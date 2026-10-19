"""Contrato que todo adapter de ``VectorStore`` precisa cumprir. Para um adapter novo, adicione-o em ``test_stores.py``."""

import numpy as np
import pytest

from workflows.apps.code_search.adapters.driven.fakes import FakeEmbedder
from workflows.apps.code_search.domain.errors import CodeSearchError
from workflows.apps.code_search.domain.model import ChunkRecord, Filters, IndexSchema, make_chunk_id
from workflows.apps.code_search.domain.sparse import JavaSparseEncoder

SCHEMA = IndexSchema(chunker_version="c1", tokenizer_version="t1", policy_version="p1")
SOURCES = {
    "a": [("src/Pagamento.java", "Pagamento.calcularJuros", "calcula juros compostos do valor"),
          ("src/Pagamento.java", "Pagamento.estornar", "estorna o pagamento da conta")],
    "b": [("src/Cliente.java", "Cliente.validarCpf", "valida o documento cpf do cliente")],
}


def build(store, repo_id, revision="r1", rows=None, embedder=None, roots=("src",)):
    embedder = embedder or FakeEmbedder(dim=8)
    rows = rows if rows is not None else SOURCES[repo_id]
    sparse = JavaSparseEncoder()
    terms = sparse.encode_documents([f"{s}\n{sym}\n{t}" for s, sym, t in rows])
    chunks = [
        ChunkRecord(make_chunk_id(embedder.profile.profile_id, repo_id, s, sym, str(i), i * 10 + 1), s, "java", sym, "",
                    i * 10 + 1, i * 10 + 5, t, tm)
        for i, ((s, sym, t), tm) in enumerate(zip(rows, terms))
    ]
    dense = embedder.embed_documents([t for _, _, t in rows])
    return store.write_segment(
        repo_id=repo_id, repo_revision=revision, indexed_at="2026-10-19T12:00:00Z",
        profile=embedder.profile, schema=SCHEMA, chunks=chunks, dense=dense, roots=roots,
    ), embedder


class StoreContract:
    @pytest.fixture
    def store(self, tmp_path):  # sobrescrito por cada adapter
        raise NotImplementedError

    def publish_all(self, store):
        refs, emb = [], FakeEmbedder(dim=8)
        for repo in ("a", "b"):
            ref, _ = build(store, repo, embedder=emb)
            refs.append(ref)
        return store.publish(profile=emb.profile, schema=SCHEMA, replace=refs), emb

    def test_empty_store_has_no_active_generation(self, store):
        assert store.open_active() is None

    def test_segment_is_invisible_until_published(self, store):
        build(store, "a")
        assert store.open_active() is None

    def test_publish_makes_generation_active_with_segment_info(self, store):
        gen, emb = self.publish_all(store)
        snap = store.open_active()
        assert snap.generation_id == gen and snap.profile.profile_id == emb.profile.profile_id
        assert snap.schema.schema_id == SCHEMA.schema_id and snap.repos == ["a", "b"]
        info = snap.segment_info("a")
        assert (info.repo_revision, info.roots, info.chunks) == ("r1", ("src",), 2)
        assert info.profile_id == emb.profile.profile_id and snap.segment_info("zzz") is None
        assert snap.coverage(["b", "zzz", "a"]) == (["b", "a"], ["zzz"])

    def test_dense_and_lexical_search_locate_and_record(self, store):
        _, emb = self.publish_all(store)
        snap = store.open_active()
        q = emb.embed_documents(["valida o documento cpf do cliente"])[0]
        out = snap.search(["a", "b"], dense_q=q, lexical_terms=["cpf"], mode="hybrid")
        assert out["dense"][0].repo_id == "b" and out["lexical"][0].repo_id == "b"
        hit = out["lexical"][0]
        loc = snap.locate(hit)
        assert (loc.source, loc.start_line, loc.end_line) == ("src/Cliente.java", 1, 5)
        rec = snap.record(hit)
        assert rec["repo"] == "b" and rec["symbol"] == "Cliente.validarCpf" and "cpf" in rec["text"]

    def test_filters_and_modes(self, store):
        _, emb = self.publish_all(store)
        snap = store.open_active()
        only_lex = snap.search(["a", "b"], lexical_terms=["juros"], mode="lexical")
        assert only_lex["dense"] == [] and only_lex["lexical"][0].repo_id == "a"
        assert snap.search(["a"], lexical_terms=["juros"], mode="lexical", filters=Filters.make(languages=["kotlin"]))["lexical"] == []
        assert snap.search(["a"], lexical_terms=["juros"], mode="lexical", filters=Filters.make(path_prefix="src"))["lexical"]
        with pytest.raises(CodeSearchError) as exc:
            snap.search(["a"], dense_q=np.ones(3, dtype=np.float32), mode="semantic")
        assert exc.value.code == "profile_mismatch"

    def test_publish_replaces_and_removes_repos_atomically(self, store):
        gen1, emb = self.publish_all(store)
        new_ref, _ = build(store, "a", revision="r2", rows=SOURCES["a"][:1], embedder=emb)
        gen2 = store.publish(profile=emb.profile, schema=SCHEMA, replace=[new_ref], remove=["b"])
        snap = store.open_active()
        assert snap.generation_id == gen2 != gen1 and snap.repos == ["a"]
        assert snap.segment_info("a").repo_revision == "r2" and snap.segment_info("a").chunks == 1

    def test_rollback_and_previous_generation(self, store):
        gen1, emb = self.publish_all(store)
        assert store.previous_generation_id() is None
        with pytest.raises(CodeSearchError):
            store.rollback()
        ref, _ = build(store, "a", revision="r2", embedder=emb)
        gen2 = store.publish(profile=emb.profile, schema=SCHEMA, replace=[ref])
        assert store.previous_generation_id() == gen1
        assert store.rollback() == gen1 and store.open_active().generation_id == gen1
        assert store.previous_generation_id() == gen2

    def test_profile_mismatch_is_rejected_and_active_generation_survives(self, store):
        gen, _ = self.publish_all(store)
        other = FakeEmbedder(dim=8, model="outro-modelo")
        ref, _ = build(store, "a", revision="r9", embedder=other)
        with pytest.raises(CodeSearchError) as exc:
            store.publish(profile=other.profile, schema=SCHEMA, replace=[ref])
        assert exc.value.code == "profile_mismatch" and store.open_active().generation_id == gen

    def test_null_vectors_are_rejected_on_write(self, store):
        emb = FakeEmbedder(dim=8)
        rows = SOURCES["b"]
        sparse = JavaSparseEncoder()
        chunk = ChunkRecord("00000000-0000-5000-8000-000000000001", rows[0][0], "java", rows[0][1], "", 1, 5, rows[0][2],
                            sparse.encode_documents([rows[0][2]])[0])
        with pytest.raises(CodeSearchError) as exc:
            store.write_segment(
                repo_id="b", repo_revision="r", indexed_at="2026-10-19T12:00:00Z", profile=emb.profile, schema=SCHEMA,
                chunks=[chunk], dense=np.zeros((1, 8), dtype=np.float32),
            )
        assert exc.value.code == "embedder_invalid_output"

    def test_prune_keeps_active_and_previous_only(self, store):
        gen1, emb = self.publish_all(store)
        ref, _ = build(store, "a", revision="r2", embedder=emb)
        store.publish(profile=emb.profile, schema=SCHEMA, replace=[ref])
        ref3, _ = build(store, "a", revision="r3", embedder=emb)
        gen3 = store.publish(profile=emb.profile, schema=SCHEMA, replace=[ref3])
        removed = store.prune()
        assert removed["generations"] >= 1
        assert store.open_active().generation_id == gen3 and store.previous_generation_id() is not None
        store.rollback()  # a anterior ainda é utilizável depois do prune
        assert store.open_active().segment_info("a").repo_revision == "r2"

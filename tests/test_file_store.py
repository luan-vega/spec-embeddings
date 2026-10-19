import json
import os
import threading

import numpy as np
import pytest

from workflows.apps.code_search import file_store
from workflows.apps.code_search.core import CodeSearchError, EmbeddingProfile, Filters
from workflows.apps.code_search.file_store import FileVectorStore
from workflows.apps.code_search.filelock import FileLock

from helpers import PROFILE, SCHEMA, basis, chunk, write


@pytest.fixture
def store(tmp_path):
    return FileVectorStore(tmp_path / "idx", lock_timeout=0.5)


def make_repo(store, repo, revision="rev1"):
    chunks = [
        chunk(repo, "src/main/java/a/Pagamento.java", "calcularJurosComposto taxa prazo", 1, symbol="Pagamento.calcular"),
        chunk(repo, "src/main/java/ab/Cliente.java", "validarCpf documento cliente", 1, symbol="Cliente.validar"),
        chunk(repo, "src/main/java/a/Conta.java", "saldo conta corrente", 1, symbol="Conta.saldo", language="kotlin"),
        chunk(repo, "src/main/java/a/Conta.java", "extrato conta periodo", 11, symbol="Conta.extrato"),
    ]
    return write(store, repo, chunks, basis(0, 1, 2, 3), revision=revision)


def test_no_active_generation_returns_none(store):
    assert store.open_active() is None


def test_write_publish_reopen_search(store, tmp_path):
    ref = make_repo(store, "svc-a")
    gen = store.publish(profile=PROFILE, schema=SCHEMA, replace=[ref])
    reopened = FileVectorStore(tmp_path / "idx").open_active()
    assert reopened.generation_id == gen and reopened.repos == ["svc-a"]

    q = np.array([0, 1, 0, 0], dtype=np.float32)
    dense = reopened.search(["svc-a"], dense_q=q, mode="semantic", depth=4)["dense"]
    assert [h.row for h in dense][0] == 1
    assert dense[0].score == pytest.approx(1.0)
    assert {h.score for h in dense[1:]} == {0.0}

    lex = reopened.search(["svc-a"], lexical_terms=["juros", "composto"], mode="lexical")["lexical"]
    assert [h.row for h in lex] == [0]
    rec = reopened.record(lex[0])
    assert rec["symbol"] == "Pagamento.calcular" and rec["repo_revision"] == "rev1"
    assert rec["source"] == "src/main/java/a/Pagamento.java" and rec["start_line"] == 1


def test_hybrid_returns_both_rankings_and_modes_are_exclusive(store):
    store.publish(profile=PROFILE, schema=SCHEMA, replace=[make_repo(store, "svc-a")])
    snap = store.open_active()
    q = basis(2)[0]
    both = snap.search(["svc-a"], dense_q=q, lexical_terms=["conta"], mode="hybrid")
    assert both["dense"][0].row == 2 and {h.row for h in both["lexical"]} == {2, 3}
    assert snap.search(["svc-a"], dense_q=q, lexical_terms=["conta"], mode="semantic")["lexical"] == []
    assert snap.search(["svc-a"], dense_q=q, lexical_terms=["conta"], mode="lexical")["dense"] == []


def test_filters_by_language_and_path_components(store):
    store.publish(profile=PROFILE, schema=SCHEMA, replace=[make_repo(store, "svc-a")])
    snap = store.open_active()
    q = np.ones(4, dtype=np.float32)

    def rows(**kw):
        return sorted(h.row for h in snap.search(["svc-a"], dense_q=q, mode="semantic", filters=Filters.make(**kw))["dense"])

    assert rows(languages=["kotlin"]) == [2]
    # "a" não casa com "ab": pertencimento por componentes, nunca startswith
    assert rows(path_prefix="src/main/java/a") == [0, 2, 3]
    assert rows(path_prefix="src/main/java/ab") == [1]
    assert rows(path_prefix="src/main/java/a", languages=["java"]) == [0, 3]
    assert rows(path_prefix="src/test") == []


def test_repos_with_similar_names_are_independent(store):
    a = write(store, "repo", [chunk("repo", "A.java", "alpha")], basis(0))
    b = write(store, "repo-old", [chunk("repo-old", "B.java", "beta")], basis(1))
    store.publish(profile=PROFILE, schema=SCHEMA, replace=[a, b])
    snap = store.open_active()
    assert snap.coverage(["repo", "repo-old", "repo-x"]) == (["repo", "repo-old"], ["repo-x"])
    hits = snap.search(["repo"], dense_q=basis(1)[0], mode="semantic")["dense"]
    assert [h.repo_id for h in hits] == ["repo"]


def test_replace_one_repo_keeps_others_and_remove_drops_repo(store):
    a1 = make_repo(store, "svc-a")
    b1 = write(store, "svc-b", [chunk("svc-b", "B.java", "beta")], basis(0), revision="b1")
    g1 = store.publish(profile=PROFILE, schema=SCHEMA, replace=[a1, b1])
    a2 = make_repo(store, "svc-a", revision="rev2")
    g2 = store.publish(profile=PROFILE, schema=SCHEMA, replace=[a2])
    snap = store.open_active()
    assert g2 != g1 and snap.repos == ["svc-a", "svc-b"]
    assert snap.segments["svc-a"].manifest["repo_revision"] == "rev2"
    assert snap.segments["svc-b"].ref == b1

    store.publish(profile=PROFILE, schema=SCHEMA, remove=["svc-b"])
    assert store.open_active().repos == ["svc-a"]


def test_rollback_and_explicit_prune(store):
    a1 = make_repo(store, "svc-a", revision="rev1")
    g1 = store.publish(profile=PROFILE, schema=SCHEMA, replace=[a1])
    a2 = make_repo(store, "svc-a", revision="rev2")
    g2 = store.publish(profile=PROFILE, schema=SCHEMA, replace=[a2])
    a3 = make_repo(store, "svc-a", revision="rev3")
    g3 = store.publish(profile=PROFILE, schema=SCHEMA, replace=[a3])

    # nada some sem prune explícito
    assert len(list((store.root / "generations").glob("*.json"))) == 3
    assert len(list((store.root / "segments").iterdir())) == 3

    assert store.rollback() == g2
    assert store.open_active().segments["svc-a"].manifest["repo_revision"] == "rev2"
    assert store.rollback() == g3  # previous passa a ser a geração desfeita

    removed = store.prune()
    assert removed["generations"] == 1 and removed["segments"] == 1
    assert store.active_generation_id() == g3
    assert store.open_active().segments["svc-a"].manifest["repo_revision"] == "rev3"
    assert g1 not in {p.stem for p in (store.root / "generations").glob("*.json")}


def test_rollback_without_previous_fails(store):
    store.publish(profile=PROFILE, schema=SCHEMA, replace=[make_repo(store, "svc-a")])
    with pytest.raises(CodeSearchError) as exc:
        store.rollback()
    assert exc.value.code == "vector_store_unavailable"


def test_publish_fails_before_commit_keeps_previous_generation(store, monkeypatch):
    g1 = store.publish(profile=PROFILE, schema=SCHEMA, replace=[make_repo(store, "svc-a")])
    a2 = make_repo(store, "svc-a", revision="rev2")
    real_replace = os.replace

    def boom(src, dst):
        if os.path.basename(str(dst)) == "active.json":
            raise OSError("disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(file_store.os, "replace", boom)
    with pytest.raises(OSError):
        store.publish(profile=PROFILE, schema=SCHEMA, replace=[a2])
    monkeypatch.undo()

    assert store.active_generation_id() == g1
    assert store.open_active().segments["svc-a"].manifest["repo_revision"] == "rev1"
    assert not list(store.root.glob("active.json.tmp-*"))


def test_segment_rename_failure_leaves_nothing_published(store, monkeypatch):
    g1 = store.publish(profile=PROFILE, schema=SCHEMA, replace=[make_repo(store, "svc-a")])
    before = sorted(p.name for p in (store.root / "segments").iterdir())

    def boom(src, dst):
        raise OSError("rename failed")

    monkeypatch.setattr(file_store.os, "rename", boom)
    with pytest.raises(OSError):
        make_repo(store, "svc-a", revision="rev2")
    monkeypatch.undo()

    assert sorted(p.name for p in (store.root / "segments").iterdir()) == before
    assert list((store.root / "tmp").iterdir()) == []
    assert store.active_generation_id() == g1


def test_failure_after_commit_still_serves_new_generation(store, monkeypatch):
    g1 = store.publish(profile=PROFILE, schema=SCHEMA, replace=[make_repo(store, "svc-a")])
    a2 = make_repo(store, "svc-a", revision="rev2")
    real_replace = os.replace

    def boom(src, dst):
        if os.path.basename(str(dst)) == "previous.json":
            raise OSError("late failure")
        return real_replace(src, dst)

    monkeypatch.setattr(file_store.os, "replace", boom)
    with pytest.raises(OSError):
        store.publish(profile=PROFILE, schema=SCHEMA, replace=[a2])
    monkeypatch.undo()

    snap = store.open_active()  # commit já aconteceu: leitores veem a geração nova, íntegra
    assert snap.generation_id != g1
    assert snap.segments["svc-a"].manifest["repo_revision"] == "rev2"


def test_profile_mismatch_is_rejected_on_publish(store):
    a = make_repo(store, "svc-a")
    store.publish(profile=PROFILE, schema=SCHEMA, replace=[a])
    other = EmbeddingProfile(model="other", digest="sha256:zzz", dim=4)
    # o segmento carregado foi construído com outro perfil
    with pytest.raises(CodeSearchError) as exc:
        store.publish(profile=other, schema=SCHEMA)
    assert exc.value.code == "profile_mismatch"
    # reindexar tudo com o novo perfil é permitido
    new = write(store, "svc-a", [chunk("svc-a", "A.java", "x", profile=other)], basis(0), profile=other)
    store.publish(profile=other, schema=SCHEMA, replace=[new])
    assert store.open_active().profile == other


def test_query_dimension_mismatch_is_profile_mismatch(store):
    store.publish(profile=PROFILE, schema=SCHEMA, replace=[make_repo(store, "svc-a")])
    snap = store.open_active()
    with pytest.raises(CodeSearchError) as exc:
        snap.search(["svc-a"], dense_q=np.ones(7, dtype=np.float32), mode="semantic")
    assert exc.value.code == "profile_mismatch"
    with pytest.raises(CodeSearchError) as exc:
        snap.search(["svc-a"], dense_q=np.zeros(4, dtype=np.float32), mode="semantic")
    assert exc.value.code == "embedder_invalid_output"


def test_zero_or_nan_vectors_are_rejected(store):
    for bad in (0.0, float("nan")):
        dense = basis(0, 1)
        dense[1, :] = bad
        with pytest.raises(CodeSearchError) as exc:
            write(store, "svc-a", [chunk("svc-a", "A.java", "a"), chunk("svc-a", "B.java", "b")], dense)
        assert exc.value.code == "embedder_invalid_output"
    assert list((store.root / "segments").iterdir()) == []


def test_invalid_repo_id_and_duplicate_ids(store):
    for bad in ("../x", "a/b", "", ".hidden", "a b"):
        with pytest.raises(CodeSearchError):
            write(store, bad, [], np.empty((0, 4), dtype=np.float32))
    c = chunk("svc-a", "A.java", "a")
    with pytest.raises(CodeSearchError):
        write(store, "svc-a", [c, c], basis(0, 1))


def test_empty_segment_is_searchable(store):
    ref = write(store, "svc-a", [], np.empty((0, 4), dtype=np.float32))
    store.publish(profile=PROFILE, schema=SCHEMA, replace=[ref])
    snap = store.open_active()
    out = snap.search(["svc-a"], dense_q=basis(0)[0], lexical_terms=["x"])
    assert out == {"dense": [], "lexical": []}


def test_tampered_segment_is_detected(store):
    ref = make_repo(store, "svc-a")
    store.publish(profile=PROFILE, schema=SCHEMA, replace=[ref])
    dense_path = store.root / "segments" / ref.segment_id / "dense.npy"
    raw = bytearray(dense_path.read_bytes())
    raw[-1] ^= 0xFF
    dense_path.write_bytes(bytes(raw))
    store.open_active()  # abertura leve só confere tamanhos
    with pytest.raises(CodeSearchError) as exc:
        store._verify_segment(dense_path.parent, ref, full=True)
    assert exc.value.code == "vector_store_unavailable"
    dense_path.write_bytes(bytes(raw)[:-8])
    with pytest.raises(CodeSearchError):
        store.open_active()


def test_corrupt_pointer_is_unavailable(store):
    (store.root / "active.json").write_text("{not json")
    with pytest.raises(CodeSearchError) as exc:
        store.open_active()
    assert exc.value.code == "vector_store_unavailable"
    assert str(store.root) not in exc.value.message


def test_writer_lock_is_exclusive(store):
    with FileLock(store.root / "write.lock", timeout=1):
        with pytest.raises(CodeSearchError) as exc:
            store.publish(profile=PROFILE, schema=SCHEMA)
        assert exc.value.code == "store_busy"
    store.publish(profile=PROFILE, schema=SCHEMA)


def test_concurrent_search_during_publication_sees_whole_generations(tmp_path):
    store = FileVectorStore(tmp_path / "idx", lock_timeout=10)
    sizes = {}

    def build(version):
        n = 3 + version % 4
        chunks = [chunk("svc-a", f"F{i}.java", f"word{i} common", 1) for i in range(n)]
        dense = np.eye(4, dtype=np.float32)[[i % 4 for i in range(n)]]
        ref = write(store, "svc-a", chunks, dense, revision=f"v{version}")
        sizes[ref.segment_id] = n
        return ref

    store.publish(profile=PROFILE, schema=SCHEMA, replace=[build(0)])
    stop = threading.Event()
    errors = []

    def reader():
        reader_store = FileVectorStore(tmp_path / "idx")
        while not stop.is_set():
            try:
                snap = reader_store.open_active()
                seg = snap.segments["svc-a"]
                assert seg.count == sizes[seg.ref.segment_id]
                out = snap.search(["svc-a"], dense_q=np.ones(4, dtype=np.float32), lexical_terms=["common"], depth=100)
                assert len(out["lexical"]) == seg.count
                snap.record(out["lexical"][-1])
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                return

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    try:
        for v in range(1, 25):
            store.publish(profile=PROFILE, schema=SCHEMA, replace=[build(v)])
    finally:
        stop.set()
        for t in threads:
            t.join(10)
    assert errors == []


def test_ids_roundtrip_and_manifest_contents(store):
    ref = make_repo(store, "svc-a", revision="abc123")
    manifest = json.loads((store.root / "segments" / ref.segment_id / "manifest.json").read_text())
    assert manifest["repo_revision"] == "abc123" and manifest["counts"]["chunks"] == 4
    assert manifest["profile_id"] == PROFILE.profile_id and manifest["schema_id"] == SCHEMA.schema_id
    ids = np.load(store.root / "segments" / ref.segment_id / "ids.npy")
    assert ids.shape == (4, 16) and ids.dtype == np.uint8

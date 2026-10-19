import numpy as np
import pytest

from workflows.apps.code_search.core import CodeSearchError
from workflows.apps.code_search.embedding_cache import EmbeddingCache
from workflows.apps.code_search.file_store import FileVectorStore
from workflows.apps.code_search.indexer import IndexService
from workflows.apps.code_search.policy import DEFAULT_POLICY, ExclusionPolicy
from workflows.apps.code_search.registry import Registry
from workflows.apps.code_search.search import SearchService
from workflows.apps.code_search.sparse import JavaSparseEncoder
from workflows.apps.code_search.testing import BagOfWordsEmbedder

from gitutil import add_symlink_entry, commit_all, git, make_repo

PAGAMENTO = '''package br.pagamento;

public class Pagamento {
    /** Calcula os juros compostos de um valor para um prazo em meses. */
    public double calcularJurosComposto(double valor, int meses) {
        return valor * Math.pow(1.02, meses);
    }

    public void estornar(String contaId) { }
}
'''
CLIENTE = '''package br.cliente;

public class Cliente {
    /** Valida o documento CPF do cliente. */
    public boolean validarCpf(String documento) {
        return documento != null && documento.length() == 11;
    }
}
'''


class CountingEmbedder(BagOfWordsEmbedder):
    def __init__(self):
        super().__init__(dim=64)
        self.embedded = 0
        self.fail = False

    def embed_documents(self, texts):
        if self.fail:
            raise CodeSearchError("embedder_unavailable", "embedding model is unavailable")
        self.embedded += len(texts)
        return super().embed_documents(texts)


@pytest.fixture
def env(tmp_path):
    repo_a = make_repo(
        tmp_path / "repos" / "svc-pagamento",
        {
            "src/main/java/br/pagamento/Pagamento.java": PAGAMENTO,
            "src/main/java/br/pagamento/.env.java": 'class Env { String password = "hunter2hunter2"; }',
            "src/main/java/br/pagamento/build/Gerado.java": "class Gerado { }",
            "src/main/java/br/pagamento/Segredo.java": 'class Segredo {\n  String t = "ghp_' + "a" * 36 + '";\n}\n',
            "src/test/java/br/pagamento/PagamentoTest.java": "class PagamentoTest { void testaJuros() { } }",
        },
    )
    add_symlink_entry(repo_a, "src/main/java/br/pagamento/Atalho.java", "Pagamento.java")
    git(repo_a, "commit", "-q", "-m", "link")
    repo_b = make_repo(tmp_path / "repos" / "svc-cliente", {"src/main/java/br/cliente/Cliente.java": CLIENTE})
    (repo_b / "src/main/java/br/cliente/SoLocal.java").write_text("class SoLocal { void naoRastreado() { } }")

    home = tmp_path / "home"
    registry = Registry(home / "repositories.json")
    registry.add("svc-pagamento", repo_a)
    registry.add("svc-cliente", repo_b)
    embedder = CountingEmbedder()
    store = FileVectorStore(home / "index")
    policy = ExclusionPolicy.from_dict(DEFAULT_POLICY)
    sparse = JavaSparseEncoder()

    def indexer(policy=policy):
        return IndexService(
            store=store, registry=registry, embedder=embedder, sparse=sparse, policy=policy,
            cache=EmbeddingCache(home / "cache", embedder.profile), batch_size=4,
            now=lambda: "2026-10-19T12:00:00Z",
        )

    search = SearchService(store=store, registry=registry, embedder=embedder, sparse=sparse)
    return type("Env", (), dict(repo_a=repo_a, repo_b=repo_b, registry=registry, embedder=embedder, store=store,
                                indexer=indexer, search=search, sparse=sparse, home=home))


def test_index_two_repos_and_search_both(env):
    report = env.indexer().index()
    assert report.status == "published" and {r.repo_id for r in report.repos} == {"svc-pagamento", "svc-cliente"}
    pag = next(r for r in report.repos if r.repo_id == "svc-pagamento")
    assert pag.revision == git(env.repo_a, "rev-parse", "HEAD")
    # só Pagamento.java entra: .env, build/, segredo, symlink e testes ficam de fora
    assert pag.files_seen == 4 and pag.files_indexed == 2 and pag.files_skipped_policy == 2
    assert pag.chunks_skipped_secret >= 1

    res = env.search.search("calcular juros compostos prazo", repos=None, top_k=5)
    assert res["status"] == "complete" and res["coverage"]["searched"] == ["svc-cliente", "svc-pagamento"]
    assert res["results"][0]["repo"] == "svc-pagamento"
    assert res["results"][0]["symbol"] == "Pagamento.calcularJurosComposto"
    assert res["index"]["generation"] == report.generation_id

    res = env.search.search("validar documento cpf", top_k=5)
    top = res["results"][0]
    assert (top["repo"], top["symbol"]) == ("svc-cliente", "Cliente.validarCpf")
    assert top["source"] == "src/main/java/br/cliente/Cliente.java"
    assert top["repo_revision"] == git(env.repo_b, "rev-parse", "HEAD") and top["indexed_at"] == "2026-10-19T12:00:00Z"
    assert "language" in top and top["start_line"] <= top["end_line"] and top["text"]

    sources = {r["source"] for q in ("Env", "Gerado", "Segredo", "Atalho", "SoLocal", "naoRastreado", "testaJuros", "hunter2hunter2")
               for r in env.search.search(q, mode="lexical")["results"]}
    assert sources <= {"src/main/java/br/pagamento/Pagamento.java", "src/main/java/br/cliente/Cliente.java"}
    assert env.search.search("Segredo", mode="lexical")["results"] == []
    assert env.search.search("naoRastreado", mode="lexical")["results"] == []
    assert "ghp_" not in str(env.search.search("Segredo token", mode="lexical"))


def test_exact_identifier_is_found_by_lexical_mode(env):
    env.indexer().index()
    res = env.search.search("estornarContaId", mode="lexical")
    assert [r["symbol"] for r in res["results"]] == ["Pagamento.estornar"]
    sem = env.search.search("calcularJurosComposto", mode="semantic", top_k=1)
    assert sem["results"][0]["repo"] in {"svc-pagamento", "svc-cliente"}  # semântico não exige identificador exato


def test_filters_and_coverage(env):
    env.indexer().index()
    res = env.search.search("cliente documento", repos=["svc-cliente", "svc-fantasma"], top_k=3)
    assert res["status"] == "partial"
    assert res["coverage"] == {"searched": ["svc-cliente"], "unavailable": [{"repo": "svc-fantasma", "reason": "unknown_repo"}]}
    assert {r["repo"] for r in res["results"]} == {"svc-cliente"}
    assert env.search.search("juros", path_prefix="src/main/java/br/cliente")["results"] != []
    assert all(r["source"].startswith("src/main/java/br/cliente/")
               for r in env.search.search("juros documento", path_prefix="src/main/java/br/cliente")["results"])
    assert env.search.search("juros", languages=["kotlin"])["results"] == []
    with pytest.raises(CodeSearchError) as exc:
        env.search.search("x", repos=["svc-fantasma"])
    assert exc.value.code == "no_repos_available"


def test_invalid_requests(env):
    env.indexer().index()
    for kw in ({"query": ""}, {"query": "x" * 5000}, {"query": "x", "mode": "fuzzy"}, {"query": "x", "top_k": 0},
               {"query": "x", "top_k": 999}, {"query": "x", "path_prefix": "../etc"}, {"query": "x", "repos": []}):
        query = kw.pop("query")
        with pytest.raises(CodeSearchError) as exc:
            env.search.search(query, **kw)
        assert exc.value.code == "invalid_request"


def test_reindex_unchanged_code_does_not_reembed(env):
    first = env.indexer().index()
    embedded = env.embedder.embedded
    assert embedded > 0 and sum(r.embeddings_new for r in first.repos) == embedded
    second = env.indexer().index()
    assert env.embedder.embedded == embedded  # cache: nada novo foi ao modelo
    assert sum(r.embeddings_new for r in second.repos) == 0
    assert sum(r.embeddings_cached for r in second.repos) >= embedded
    assert second.generation_id != first.generation_id


def test_changed_file_embeds_only_changed_chunks(env):
    env.indexer().index()
    before = env.embedder.embedded
    path = env.repo_a / "src/main/java/br/pagamento/Pagamento.java"
    path.write_text(PAGAMENTO.replace("1.02", "1.03"), encoding="utf-8", newline="\n")
    new_rev = commit_all(env.repo_a)
    report = env.indexer().index()
    assert env.embedder.embedded - before == 1  # só o método alterado
    pag = next(r for r in report.repos if r.repo_id == "svc-pagamento")
    assert pag.revision == new_rev
    assert env.search.search("juros compostos", repos=["svc-pagamento"])["results"][0]["repo_revision"] == new_rev


def test_embedding_failure_preserves_active_generation(env):
    first = env.indexer().index()
    path = env.repo_b / "src/main/java/br/cliente/Cliente.java"
    path.write_text(CLIENTE.replace("11", "14"), encoding="utf-8", newline="\n")
    commit_all(env.repo_b)
    env.embedder.fail = True
    with pytest.raises(CodeSearchError) as exc:
        env.indexer().index()
    assert exc.value.code == "embedder_unavailable"
    assert env.store.active_generation_id() == first.generation_id
    res = env.search.search("validar documento cpf", repos=["svc-cliente"])
    assert res["index"]["generation"] == first.generation_id
    assert "== 11" in res["results"][0]["text"]
    env.embedder.fail = False
    assert env.indexer().index().generation_id != first.generation_id


def test_null_vectors_from_embedder_are_rejected(env):
    class ZeroEmbedder(CountingEmbedder):
        def embed_documents(self, texts):
            return np.zeros((len(texts), self.profile.dim), dtype=np.float32)

    env.embedder.embed_documents = ZeroEmbedder().embed_documents
    with pytest.raises(CodeSearchError) as exc:
        env.indexer().index()
    assert exc.value.code == "embedder_invalid_output"
    assert env.store.active_generation_id() is None


def test_missing_or_invalid_policy_indexes_nothing(env, tmp_path):
    for policy in (ExclusionPolicy.load(tmp_path / "ausente.json"), ExclusionPolicy.from_dict({"version": "1"})):
        report = env.indexer(policy).index()
        assert report.status == "skipped" and report.reason.startswith("policy_")
    assert env.store.active_generation_id() is None
    assert env.embedder.embedded == 0


def test_disabled_repo_leaves_generation_but_keeps_snapshots(env):
    env.indexer().index()
    segments = sorted(p.name for p in (env.home / "index" / "segments").iterdir())
    env.registry.set_enabled("svc-cliente", False)
    report = env.indexer().index()
    assert report.removed == ["svc-cliente"]
    snap = env.store.open_active()
    assert snap.repos == ["svc-pagamento"]
    assert set(segments) <= {p.name for p in (env.home / "index" / "segments").iterdir()}  # nada foi apagado
    res = env.search.search("documento", repos=["svc-cliente", "svc-pagamento"])
    assert res["coverage"]["unavailable"] == [{"repo": "svc-cliente", "reason": "disabled"}]
    env.registry.remove("svc-cliente")
    assert env.search.search("documento", repos=["svc-cliente", "svc-pagamento"])["coverage"]["unavailable"] == [
        {"repo": "svc-cliente", "reason": "unknown_repo"}
    ]


def test_profile_mismatch_between_index_and_runtime(env):
    env.indexer().index()
    other = BagOfWordsEmbedder(dim=64, model="outro-modelo")
    service = SearchService(store=env.store, registry=env.registry, embedder=other, sparse=env.sparse)
    with pytest.raises(CodeSearchError) as exc:
        service.search("documento")
    assert exc.value.code == "profile_mismatch" and exc.value.http_status == 409


def test_search_before_any_index_has_no_repos(env):
    with pytest.raises(CodeSearchError) as exc:
        env.search.search("qualquer coisa")
    assert exc.value.code == "no_repos_available"


def test_overlapping_chunks_are_collapsed(env, tmp_path):
    body = "\n".join(f"        total += item{i} * taxa; // etapa {i}" for i in range(120))
    repo = make_repo(tmp_path / "repos" / "svc-grande",
                     {"src/main/java/G.java": f"class G {{\n    void processar() {{\n{body}\n    }}\n}}\n"})
    env.registry.add("svc-grande", repo)
    env.indexer().index(["svc-grande"])
    res = env.search.search("total taxa etapa processar", repos=["svc-grande"], top_k=10)
    spans = [(r["start_line"], r["end_line"]) for r in res["results"]]
    assert len(spans) >= 1
    for i, (s1, e1) in enumerate(spans):
        for s2, e2 in spans[i + 1:]:
            assert e1 < s2 or e2 < s1

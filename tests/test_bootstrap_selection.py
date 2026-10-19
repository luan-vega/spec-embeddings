import pytest

from workflows.apps.code_search import bootstrap
from workflows.apps.code_search.adapters.driven.fakes import FakeEmbedder
from workflows.apps.code_search.adapters.driven.file_store import FileVectorStore
from workflows.apps.code_search.bootstrap import build_runtime, load_embedder, load_store, register_embedder, register_store
from workflows.apps.code_search.config import Paths
from workflows.apps.code_search.domain.errors import CodeSearchError


@pytest.fixture(autouse=True)
def restore_registries(monkeypatch):
    monkeypatch.setattr(bootstrap, "EMBEDDERS", dict(bootstrap.EMBEDDERS))
    monkeypatch.setattr(bootstrap, "STORES", dict(bootstrap.STORES))


def test_defaults_are_ollama_and_file_store(tmp_path):
    with pytest.raises(CodeSearchError) as exc:  # Ollama é o padrão e exige o modelo
        load_embedder({})
    assert exc.value.code == "embedder_unavailable" and "CODE_SEARCH_OLLAMA_MODEL" in exc.value.message
    assert isinstance(load_store(Paths(tmp_path), {}), FileVectorStore)


def test_embedder_is_selected_by_configuration():
    assert load_embedder({"CODE_SEARCH_EMBEDDER": "fake"}).profile.model
    assert load_embedder({"CODE_SEARCH_DEV_FAKE_EMBEDDER": "1"}).profile.model  # variável antiga continua valendo
    with pytest.raises(CodeSearchError) as exc:
        load_embedder({"CODE_SEARCH_EMBEDDER": "nao-existe"})
    assert exc.value.code == "embedder_unavailable" and "ollama" in exc.value.message


def test_registered_embedder_can_replace_ollama_without_touching_the_core():
    register_embedder("meu-modelo", lambda env: FakeEmbedder(dim=4, model=env.get("M", "x")))
    emb = load_embedder({"CODE_SEARCH_EMBEDDER": "meu-modelo", "M": "abc"})
    assert (emb.profile.model, emb.profile.dim) == ("abc", 4)


def test_registered_store_is_used_by_the_runtime(tmp_path, monkeypatch):
    created = []

    def factory(paths, env):
        created.append(paths.home)
        return FileVectorStore(paths.home / "outro-banco")

    register_store("outro", factory)
    monkeypatch.setenv("CODE_SEARCH_STORE", "outro")
    rt = build_runtime(tmp_path, with_embedder=False)
    assert created == [tmp_path] and (tmp_path / "outro-banco").exists() and rt.store is not None
    with pytest.raises(CodeSearchError) as exc:
        load_store(Paths(tmp_path), {"CODE_SEARCH_STORE": "nada"})
    assert exc.value.code == "vector_store_unavailable"


def test_non_local_adapters_are_opt_in(tmp_path):
    register_embedder("nuvem", lambda env: FakeEmbedder(), local=False)
    register_store("banco-remoto", lambda paths, env: FileVectorStore(paths.home / "x"), local=False)
    with pytest.raises(CodeSearchError) as exc:
        load_embedder({"CODE_SEARCH_EMBEDDER": "nuvem"})
    assert "CODE_SEARCH_ALLOW_REMOTE" in exc.value.message
    with pytest.raises(CodeSearchError):
        load_store(Paths(tmp_path), {"CODE_SEARCH_STORE": "banco-remoto"})
    assert load_embedder({"CODE_SEARCH_EMBEDDER": "nuvem", "CODE_SEARCH_ALLOW_REMOTE": "1"}) is not None
    assert load_store(Paths(tmp_path), {"CODE_SEARCH_STORE": "banco-remoto", "CODE_SEARCH_ALLOW_REMOTE": "1"})

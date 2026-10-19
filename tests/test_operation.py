import io
import json

import pytest

from workflows.apps.code_search.adapters.driving import cli
from workflows.apps.code_search.adapters.driving.cli import main

from gitutil import commit_all, make_repo
from rtutil import CLIENTE, PAGAMENTO


def run(home, *argv, monkeypatch):
    monkeypatch.setenv("CODE_SEARCH_DEV_FAKE_EMBEDDER", "1")
    out, err = io.StringIO(), io.StringIO()
    code = main(["--home", str(home), *argv], stdout=out, stderr=err, stdin=io.StringIO(""))
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    home = tmp_path / "home"
    a = make_repo(tmp_path / "a", {"src/main/java/br/Pagamento.java": PAGAMENTO})
    b = make_repo(tmp_path / "b", {"src/main/java/br/Cliente.java": CLIENTE})
    cmd = lambda *argv: run(home, *argv, monkeypatch=monkeypatch)
    assert cmd("policy", "init")[0] == 0
    assert cmd("repo", "add", "svc-a", str(a))[0] == 0
    assert cmd("repo", "add", "svc-b", str(b))[0] == 0
    return cmd, home, a, b


def commit_change(repo):
    path = repo / "src/main/java/br/Extra.java"
    path.write_text("package br;\npublic class Extra { public int dobro(int x) { return x * 2; } }\n")
    commit_all(repo, "extra")


def test_reindex_skips_unchanged_repos_and_updates_only_the_changed_one(setup):
    cmd, home, a, b = setup
    assert cmd("index")[0] == 0
    code, out, _ = cmd("index")
    assert code == 0 and "nada mudou" in out
    commit_change(a)
    code, out, _ = cmd("index", "--repo", "svc-a")
    assert code == 0 and "svc-a: " in out and "sem mudanças" not in out and "ativa" in out
    code, out, _ = cmd("index")
    assert code == 0 and "nada mudou" in out
    code, out, _ = cmd("search", "dobro", "--mode", "lexical", "--json")
    assert json.loads(out)["results"][0]["symbol"].endswith("dobro")


def test_changing_roots_forces_reindex(setup):
    cmd, home, a, b = setup
    cmd("index")
    registry = home / "repositories.json"
    data = json.loads(registry.read_text())
    for e in data["repositories"]:
        if e["repo_id"] == "svc-a":
            e["roots"] = ["src"]
    registry.write_text(json.dumps(data))
    code, out, _ = cmd("index")
    assert code == 0 and "nada mudou" not in out


def test_index_force_republishes(setup):
    cmd, *_ = setup
    cmd("index")
    code, out, _ = cmd("index", "--force")
    assert code == 0 and "do cache" in out


def test_rollback_and_prune(setup):
    cmd, home, a, b = setup
    cmd("index")
    first = json.loads(cmd("status", "--json")[1])["index"]["generation"]
    commit_change(a)
    cmd("index")
    status = json.loads(cmd("status", "--json")[1])["index"]
    assert status["previous_generation"] == first and status["generation"] != first
    code, out, _ = cmd("rollback")
    assert code == 0 and first in out
    assert json.loads(cmd("status", "--json")[1])["index"]["generation"] == first
    assert "anterior" in cmd("status")[1]
    code, out, _ = cmd("prune")
    assert code == 0 and "removidos" in out
    assert cmd("search", "calcular juros", "--mode", "lexical")[0] == 0  # índice segue íntegro


def test_rollback_without_previous_is_explicit(setup):
    cmd, *_ = setup
    cmd("index")
    code, _, err = cmd("rollback")
    assert code == 1 and "erro [vector_store_unavailable]" in err


def test_unexpected_failure_shows_no_traceback_or_paths(setup, monkeypatch):
    cmd, home, *_ = setup

    def boom(*a, **k):
        raise RuntimeError(f"falha em {home}/segredo.java")

    monkeypatch.setattr(cli, "_dispatch", boom)
    code, out, err = cmd("status")
    assert code == 1 and err == "erro [internal_error]: unexpected failure\n" and str(home) not in out + err

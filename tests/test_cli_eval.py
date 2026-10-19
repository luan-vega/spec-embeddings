import io
import json

import pytest

from workflows.apps.code_search.application import evaluation as ev
from workflows.apps.code_search.adapters.driving.cli import main
from workflows.apps.code_search.bootstrap import build_runtime
from workflows.apps.code_search.adapters.driven.fakes import BagOfWordsEmbedder

from gitutil import make_repo
from rtutil import CLIENTE, PAGAMENTO, make_indexed_runtime


def run_cli(*argv, home=None, env_fake=True, monkeypatch=None, stdin=""):
    out, err = io.StringIO(), io.StringIO()
    if monkeypatch is not None:
        if env_fake:
            monkeypatch.setenv("CODE_SEARCH_DEV_FAKE_EMBEDDER", "1")
        else:
            monkeypatch.delenv("CODE_SEARCH_DEV_FAKE_EMBEDDER", raising=False)
    code = main(["--home", str(home), *argv], stdout=out, stderr=err, stdin=io.StringIO(stdin))
    return code, out.getvalue(), err.getvalue()


def test_full_cli_flow(tmp_path, monkeypatch):
    home = tmp_path / "home"
    repo_a = make_repo(tmp_path / "a", {"src/main/java/br/Pagamento.java": PAGAMENTO})
    repo_b = make_repo(tmp_path / "b", {"src/main/java/br/Cliente.java": CLIENTE})
    cli = lambda *a, **k: run_cli(*a, home=home, monkeypatch=monkeypatch, **k)

    assert cli("repo", "list")[1] == ""
    assert cli("repo", "add", "svc-a", str(repo_a))[0] == 0
    assert cli("repo", "add", "svc-b", str(repo_b))[0] == 0
    code, out, err = cli("repo", "add", "svc-a", str(repo_a))
    assert code == 1 and "erro [invalid_request]" in err
    assert [l.split("\t")[0] for l in cli("repo", "list")[1].splitlines()] == ["svc-a", "svc-b"]

    code, out, _ = cli("index")  # política ausente: indexa nada
    assert code == 1 and "policy_missing" in out
    assert cli("policy", "init")[0] == 0
    assert cli("policy", "init")[0] == 1 and cli("policy", "init", "--force")[0] == 0

    code, out, _ = cli("index")
    assert code == 0 and "svc-a:" in out and "geração" in out
    code, out, _ = cli("search", "calcular juros compostos", "--top-k", "2")
    assert code == 0 and "Pagamento.calcularJurosComposto" in out and "complete" in out
    code, out, _ = cli("search", "validarCpf", "--mode", "lexical", "--json", "--repo", "svc-b")
    payload = json.loads(out)
    assert code == 0 and payload["results"][0]["symbol"] == "Cliente.validarCpf"
    code, out, _ = cli("status", "--json")
    status = json.loads(out)
    assert code == 0 and all(r["indexed"] for r in status["repos"]) and status["policy"]["valid"]
    assert str(tmp_path) not in out
    code, out, _ = cli("status")
    assert "indexado" in out

    code, out, _ = cli("repo", "remove", "svc-b")
    assert code == 0 and "snapshots não são apagados" in out
    assert (home / "index" / "segments").exists() and len(list((home / "index" / "segments").iterdir())) == 2
    assert cli("repo", "remove", "svc-b")[0] == 1


def test_embedder_failure_is_explicit_and_never_silent(tmp_path, monkeypatch):
    code, out, err = run_cli("search", "x", home=tmp_path / "h", monkeypatch=monkeypatch, env_fake=False)
    assert code == 1 and "erro [embedder_unavailable]" in err
    code, _, err = run_cli("ollama-probe", home=tmp_path / "h", monkeypatch=monkeypatch, env_fake=False)
    assert code == 1 and "embedder_unavailable" in err
    # comandos que não precisam do modelo continuam funcionando
    assert run_cli("repo", "list", home=tmp_path / "h", monkeypatch=monkeypatch, env_fake=False)[0] == 0


def test_cli_search_without_index_reports_no_repos(tmp_path, monkeypatch):
    home = tmp_path / "home"
    repo = make_repo(tmp_path / "a", {"src/main/java/A.java": "class A {}"})
    run_cli("repo", "add", "svc-a", str(repo), home=home, monkeypatch=monkeypatch)
    code, _, err = run_cli("search", "algo", home=home, monkeypatch=monkeypatch)
    assert code == 1 and "no_repos_available" in err


# ------------------------------------------------------------------ métricas

def test_metrics_known_values():
    flags = [False, True, False, True]
    assert ev.recall_at_k(flags, 2, 1) == 0.0 and ev.recall_at_k(flags, 2, 2) == 0.5 and ev.recall_at_k(flags, 2, 4) == 1.0
    assert ev.reciprocal_rank(flags) == 0.5 and ev.reciprocal_rank([False]) == 0.0
    assert ev.ndcg_at_k([True, False], 1, 10) == 1.0
    assert round(ev.ndcg_at_k([False, True], 1, 10), 4) == 0.6309
    assert ev.ndcg_at_k([], 0, 10) == 0.0
    assert ev.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5 and ev.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert ev.percentile([], 95) == 0.0


def test_dataset_loading_rejects_bad_lines(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"id":"a","uc":"UC1","query":"q","relevant":[{"repo":"r","source":"s"}]}\n\n')
    assert len(ev.load_dataset(p)) == 1
    p.write_text('{"id":"a","uc":"UC1"}\n')
    with pytest.raises(ValueError):
        ev.load_dataset(p)
    p.write_text('{"id":"a","uc":"UC1","query":"q","relevant":[]}\n')
    with pytest.raises(ValueError):
        ev.load_dataset(p)


def test_evaluate_and_thresholds_over_real_pipeline(tmp_path, monkeypatch):
    rt = make_indexed_runtime(tmp_path)
    cases = [
        ev.Case("c1", "UC1", "calcular juros compostos", ({"repo": "svc-pagamento", "source": "src/main/java/br/pagamento/Pagamento.java"},)),
        ev.Case("c2", "UC2", "estornar", ({"repo": "svc-pagamento", "source": "src/main/java/br/pagamento/Pagamento.java", "symbol": "Pagamento.estornar"},)),
        ev.Case("c3", "UC2", "validarCpf", ({"repo": "svc-cliente", "source": "src/main/java/br/cliente/Cliente.java", "symbol": "Cliente.validarCpf"},)),
    ]
    report = ev.evaluate(rt.search.search, cases, k=10)
    assert set(report["modes"]) == {"hybrid", "semantic", "lexical"}
    lex = report["modes"]["lexical"]["UC2"]
    assert lex["recall@5"] == 1.0 and lex["mrr"] == 1.0 and lex["n"] == 2
    assert report["modes"]["hybrid"]["UC1"]["latency_p95_ms"] >= 0

    ok = {"rules": [{"uc": "UC2", "mode": "lexical", "metric": "recall@5", "min": 0.95},
                    {"uc": "UC2", "mode": "hybrid", "metric": "latency_p95_ms", "max": 5000}]}
    assert ev.check_thresholds(report, ok) == []
    bad = {"rules": [{"uc": "UC2", "mode": "lexical", "metric": "recall@5", "min": 1.1},
                     {"uc": "UC9", "mode": "hybrid", "metric": "mrr", "min": 0.1},
                     {"uc": "UC1", "mode": "hybrid", "metric": "ndcg@10", "min_delta_vs": "lexical", "delta": 5.0}]}
    failures = ev.check_thresholds(report, bad)
    assert len(failures) == 3 and failures[1]["reason"] == "missing"


def test_eval_command_records_threshold_hash_and_exit_code(tmp_path, monkeypatch):
    home = tmp_path / "home"
    repo = make_repo(tmp_path / "a", {"src/main/java/br/Cliente.java": CLIENTE})
    cli = lambda *a: run_cli(*a, home=home, monkeypatch=monkeypatch)
    cli("repo", "add", "svc-b", str(repo))
    cli("policy", "init")
    cli("index")
    dataset = tmp_path / "d.jsonl"
    dataset.write_text(json.dumps({"id": "c", "uc": "UC2", "query": "validarCpf",
                                   "relevant": [{"repo": "svc-b", "source": "src/main/java/br/Cliente.java"}]}) + "\n")
    thr = tmp_path / "t.json"
    thr.write_text(json.dumps({"k": 10, "rules": [{"uc": "UC2", "mode": "lexical", "metric": "recall@5", "min": 0.9}]}))
    out_file = tmp_path / "report.json"
    code, out, _ = cli("eval", "--dataset", str(dataset), "--thresholds", str(thr), "--out", str(out_file))
    report = json.loads(out_file.read_text())
    assert code == 0 and report["passed"] is True and len(report["thresholds_sha256"]) == 64
    thr.write_text(json.dumps({"k": 10, "rules": [{"uc": "UC2", "mode": "lexical", "metric": "recall@5", "min": 2}]}))
    code, out, _ = cli("eval", "--dataset", str(dataset), "--thresholds", str(thr))
    assert code == 1 and json.loads(out)["passed"] is False

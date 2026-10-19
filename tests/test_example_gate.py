import importlib.util
import io
import json
import sys
from pathlib import Path

from workflows.apps.code_search.adapters.driving.cli import main

EXAMPLE = Path(__file__).resolve().parents[1] / "eval" / "example"


def load_builder():
    sys.path.insert(0, str(EXAMPLE))
    try:
        spec = importlib.util.spec_from_file_location("build_example", EXAMPLE / "build_example.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE))


def test_committed_dataset_matches_the_generator():
    builder = load_builder()
    assert (EXAMPLE / "dataset.jsonl").read_text(encoding="utf-8") == builder.dataset_text()
    cases = [json.loads(l) for l in builder.dataset_text().splitlines()]
    assert {c["uc"] for c in cases} == {"UC1", "UC2", "UC3"}
    assert all(sum(c["uc"] == uc for c in cases) >= 30 for uc in ("UC1", "UC2", "UC3"))


def test_example_corpus_runs_through_the_gate_end_to_end(tmp_path, monkeypatch):
    builder = load_builder()
    dataset = builder.build(tmp_path / "ex")
    home = tmp_path / "home"
    monkeypatch.setenv("CODE_SEARCH_DEV_FAKE_EMBEDDER", "1")

    def cli(*argv):
        out, err = io.StringIO(), io.StringIO()
        code = main(["--home", str(home), *argv], stdout=out, stderr=err, stdin=io.StringIO(""))
        return code, out.getvalue(), err.getvalue()

    assert cli("policy", "init")[0] == 0
    for repo in ("svc-pagamento", "svc-cliente", "svc-estoque"):
        assert cli("repo", "add", repo, str(tmp_path / "ex" / "repos" / repo))[0] == 0
    code, out, _ = cli("index")
    assert code == 0 and out.count(" chunks") == 3
    report_path = tmp_path / "report.json"
    thresholds = EXAMPLE.parent / "thresholds.json"
    code, _, _ = cli("eval", "--dataset", str(dataset), "--thresholds", str(thresholds), "--out", str(report_path))
    report = json.loads(report_path.read_text())
    assert report["cases"] == 108 and set(report["modes"]) == {"hybrid", "semantic", "lexical"}
    # com o embedder falso (bag of words) o resultado não mede qualidade semântica; só garante que o harness roda
    assert report["modes"]["lexical"]["UC2"]["recall@5"] == 1.0
    assert report["modes"]["hybrid"]["UC2"]["recall@5"] >= 0.95
    assert len(report["thresholds_sha256"]) == 64 and code in (0, 1)

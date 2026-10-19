import http.client
import json
import threading
import time

import pytest

from workflows.apps.code_search.api.server import CodeSearchServer
from workflows.apps.code_search.core import CodeSearchError
from workflows.apps.code_search.testing import BagOfWordsEmbedder

from rtutil import make_indexed_runtime
from test_limits_logs import capture_logs


class Client:
    def __init__(self, server):
        self.port = server.server_address[1]

    def request(self, method, path, body=None, headers=None, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        hdrs = {"Content-Type": "application/json", **(headers or {})}
        payload = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        conn.request(method, path, body=payload, headers=hdrs)
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, dict(resp.getheaders()), json.loads(data) if data else None


@pytest.fixture
def served(tmp_path):
    started = []

    def start(rt):
        server = CodeSearchServer(rt, "127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append(server)
        return Client(server)

    yield start, tmp_path
    for s in started:
        s.shutdown()
        s.server_close()


def test_health_ready_and_search_contract(served):
    start, tmp_path = served
    rt = make_indexed_runtime(tmp_path, index=False)
    client = start(rt)
    assert client.request("GET", "/health")[:1] == (200,) and client.request("GET", "/health")[2] == {"status": "ok"}
    status, _, body = client.request("GET", "/ready")
    assert status == 503 and body["ready"] is False and body["checks"]["index"] == "no_index"

    rt.indexer.index()
    status, _, body = client.request("GET", "/ready")
    assert status == 200 and body == {"ready": True, "checks": {"embedder": "ok", "index": "ok", "profile": "ok"}}

    status, headers, body = client.request(
        "POST", "/v1/search", {"query": "calcular juros compostos", "mode": "hybrid", "top_k": 3}
    )
    assert status == 200 and headers["Content-Type"].startswith("application/json")
    assert body["status"] == "complete" and body["request_id"]
    assert set(body) == {"status", "results", "coverage", "index", "request_id"}
    assert body["coverage"] == {"searched": ["svc-cliente", "svc-pagamento"], "unavailable": []}
    assert set(body["index"]) == {"profile_id", "generation"}
    first = body["results"][0]
    assert set(first) == {"repo", "source", "symbol", "parent_symbol", "start_line", "end_line", "language",
                          "score", "text", "repo_revision", "indexed_at"}
    assert first["symbol"] == "Pagamento.calcularJurosComposto"


def test_filters_and_partial_status_over_http(served):
    start, tmp_path = served
    client = start(make_indexed_runtime(tmp_path))
    status, _, body = client.request("POST", "/v1/search", {
        "query": "documento", "repos": ["svc-cliente", "svc-x"], "languages": ["java"],
        "path_prefix": "src/main/java/br/cliente", "mode": "lexical",
    })
    assert status == 200 and body["status"] == "partial"
    assert body["coverage"]["unavailable"] == [{"repo": "svc-x", "reason": "unknown_repo"}]
    assert {r["repo"] for r in body["results"]} == {"svc-cliente"}


@pytest.mark.parametrize(
    "kwargs,status,code",
    [
        ({"body": {"query": ""}}, 400, "invalid_request"),
        ({"body": {"query": "x", "mode": "fuzzy"}}, 400, "invalid_request"),
        ({"body": {"query": "x", "top_k": 1000}}, 400, "invalid_request"),
        ({"body": {"query": "x", "top_k": "3"}}, 400, "invalid_request"),
        ({"body": {"query": "x", "extra": 1}}, 400, "invalid_request"),
        ({"body": {"query": "x", "repos": "svc"}}, 400, "invalid_request"),
        ({"body": {"query": "x", "path_prefix": "../etc"}}, 400, "invalid_request"),
        ({"body": ["x"]}, 400, "invalid_request"),
        ({"raw": b"{nao json"}, 400, "invalid_request"),
        ({"body": {"query": "x"}, "headers": {"Content-Type": "text/plain"}}, 400, "invalid_request"),
        ({"body": {"query": "x"}, "headers": {"Host": "evil.example.com"}}, 400, "invalid_request"),
        ({"body": {"query": "x", "repos": ["nada"]}}, 404, "no_repos_available"),
        ({"raw": b"{" + b" " * (70 * 1024) + b"}"}, 413, "payload_too_large"),
    ],
)
def test_error_contract_has_no_internal_details(served, kwargs, status, code):
    start, tmp_path = served
    client = start(make_indexed_runtime(tmp_path))
    got, _, body = client.request("POST", "/v1/search", **kwargs)
    assert got == status
    assert set(body) == {"error"} and set(body["error"]) == {"code", "message", "request_id"}
    assert body["error"]["code"] == code and body["error"]["request_id"]
    assert str(tmp_path) not in json.dumps(body)


def test_routes_and_methods(served):
    start, tmp_path = served
    client = start(make_indexed_runtime(tmp_path))
    assert client.request("GET", "/nada")[0] == 404
    assert client.request("POST", "/v1/outra", {"query": "x"})[0] == 404
    assert client.request("PUT", "/v1/search", {"query": "x"})[0] == 405
    assert client.request("DELETE", "/v1/search")[0] == 405


def test_profile_mismatch_is_409(served):
    start, tmp_path = served
    make_indexed_runtime(tmp_path)  # índice criado com o perfil padrão
    from workflows.apps.code_search.runtime import build_runtime

    other = build_runtime(tmp_path / "home", embedder=BagOfWordsEmbedder(model="outro-modelo"))
    client = start(other)
    status, _, body = client.request("POST", "/v1/search", {"query": "juros"})
    assert status == 409 and body["error"]["code"] == "profile_mismatch"
    assert client.request("GET", "/ready")[0] == 503


def test_overload_returns_503_with_retry_after_and_health_still_answers(served):
    start, tmp_path = served
    rt = make_indexed_runtime(tmp_path, max_concurrent=1)
    real = rt.search
    gate, entered = threading.Event(), threading.Event()

    class Blocking:
        def search(self, *a, **kw):
            entered.set()
            gate.wait(10)
            return real.search(*a, **kw)

    rt.search = Blocking()
    client = start(rt)
    first = []
    t = threading.Thread(target=lambda: first.append(client.request("POST", "/v1/search", {"query": "juros"})))
    t.start()
    assert entered.wait(5)
    status, headers, body = client.request("POST", "/v1/search", {"query": "juros"})
    assert status == 503 and body["error"]["code"] == "overloaded" and headers["Retry-After"] == "1"
    assert client.request("GET", "/health")[0] == 200  # saúde responde durante a carga
    gate.set()
    t.join(10)
    assert first[0][0] == 200


def test_timeout_is_504(served):
    start, tmp_path = served
    rt = make_indexed_runtime(tmp_path, timeout_s=0.2)

    class Slow:
        def search(self, *a, **kw):
            time.sleep(1.0)

    rt.search = Slow()
    status, _, body = start(rt).request("POST", "/v1/search", {"query": "juros"})
    assert status == 504 and body["error"]["code"] == "timeout"


def test_only_loopback_can_bind(tmp_path):
    rt = make_indexed_runtime(tmp_path, index=False)
    for host in ("0.0.0.0", "192.168.0.10", "example.com"):
        with pytest.raises(CodeSearchError):
            CodeSearchServer(rt, host, 0)


def test_logs_have_stages_and_no_query_text_or_paths(served):
    start, tmp_path = served
    client = start(make_indexed_runtime(tmp_path))
    stream, stop = capture_logs()
    try:
        status, _, body = client.request("POST", "/v1/search", {"query": "token super-sensivel /home/luan/repo", "mode": "lexical"})
        assert status == 200
        client.request("POST", "/v1/search", {"query": ""})
    finally:
        stop()
    lines = [json.loads(l) for l in stream.getvalue().splitlines()]
    ok, bad = lines
    assert ok["event"] == "search" and ok["status"] == "complete" and ok["mode"] == "lexical"
    assert ok["request_id"] == body["request_id"] and ok["profile_id"] and ok["generation"]
    assert set(ok["stages_ms"]) == {"resolve", "embed", "retrieve", "fuse", "hydrate"} and ok["duration_ms"] >= 0
    assert "repos_searched" in ok and "candidates" in ok and "results" in ok
    assert bad["status"] == "error" and bad["code"] == "invalid_request"
    text = stream.getvalue()
    assert "super-sensivel" not in text and "/home/luan" not in text and str(tmp_path) not in text
    assert "text" not in ok and "query" not in ok

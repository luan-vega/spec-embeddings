import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from workflows.apps.code_search.core import CodeSearchError
from workflows.apps.code_search.ollama import OllamaClient, OllamaEmbedder
from workflows.apps.code_search.runtime import load_embedder


class FakeOllama:
    def __init__(self, dim=4, models=("nomic-embed-text:latest",), embed_status=200, embeddings=None):
        self.dim, self.models, self.embed_status, self.embeddings = dim, models, embed_status, embeddings
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status, body):
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                outer.requests.append(("GET", self.path, None))
                self._send(200, {"models": [{"name": m, "digest": "abc123"} for m in outer.models]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(("POST", self.path, body))
                if self.path == "/api/show":
                    return self._send(200, {"model_info": {"bert.context_length": 2048}})
                if outer.embed_status != 200:
                    return self._send(outer.embed_status, {"error": "x"})
                vecs = outer.embeddings or [[float(i + 1)] * outer.dim for i in range(len(body["input"]))]
                self._send(200, {"embeddings": vecs})

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def ollama():
    made = []

    def make(**kw):
        f = FakeOllama(**kw)
        made.append(f)
        return f

    yield make
    for f in made:
        f.close()


def test_profile_from_probe(ollama):
    f = ollama(dim=6)
    e = OllamaEmbedder("nomic-embed-text", url=f.url, doc_prefix="d: ", query_prefix="q: ")
    p = e.profile
    assert (p.model, p.digest, p.dim, p.max_input_tokens) == ("nomic-embed-text", "sha256:abc123", 6, 2048)
    assert e.ready()


def test_embeds_normalized_without_truncation_and_with_prefix(ollama):
    f = ollama()
    e = OllamaEmbedder("nomic-embed-text", url=f.url, doc_prefix="d: ", query_prefix="q: ")
    docs = e.embed_documents(["a", "b"])
    assert docs.dtype == np.float32 and np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-6)
    e.embed_query("x")
    posts = [b for m, p, b in f.requests if p == "/api/embed"]
    assert all(b["truncate"] is False for b in posts)
    assert posts[-2]["input"] == ["d: a", "d: b"] and posts[-1]["input"] == ["q: x"]


def test_batches_large_inputs(ollama):
    f = ollama()
    e = OllamaEmbedder("nomic-embed-text", url=f.url)
    assert e.embed_documents([str(i) for i in range(40)]).shape == (40, 4)


def test_model_not_installed_never_pulls(ollama):
    f = ollama(models=("other:latest",))
    with pytest.raises(CodeSearchError) as exc:
        OllamaEmbedder("nomic-embed-text", url=f.url)
    assert exc.value.code == "embedder_unavailable"
    assert not any(p == "/api/pull" for _, p, _ in f.requests)


def test_unreachable_is_embedder_unavailable():
    with pytest.raises(CodeSearchError) as exc:
        OllamaEmbedder("m", url="http://127.0.0.1:1")
    assert exc.value.code == "embedder_unavailable"


@pytest.mark.parametrize("url", ["http://example.com:11434", "https://127.0.0.1:11434", "http://10.0.0.5:11434"])
def test_refuses_non_loopback(url):
    with pytest.raises(CodeSearchError) as exc:
        OllamaClient(url)
    assert exc.value.code == "embedder_unavailable"


def test_invalid_vectors_rejected(ollama):
    f = ollama()
    e = OllamaEmbedder("nomic-embed-text", url=f.url)
    f.embeddings = [[0.0, 0.0, 0.0, 0.0]]
    with pytest.raises(CodeSearchError) as exc:
        e.embed_query("x")
    assert exc.value.code == "embedder_invalid_output"
    f.embeddings = [[1.0, 2.0]]
    with pytest.raises(CodeSearchError):
        e.embed_query("x")


def test_too_long_input_is_not_truncated_silently(ollama):
    f = ollama()
    e = OllamaEmbedder("nomic-embed-text", url=f.url)
    f.embed_status = 400
    with pytest.raises(CodeSearchError) as exc:
        e.embed_query("x" * 10)
    assert exc.value.code == "embedder_invalid_output"


def test_load_embedder_requires_model_and_uses_env(ollama):
    with pytest.raises(CodeSearchError) as exc:
        load_embedder({})
    assert exc.value.code == "embedder_unavailable"
    f = ollama()
    e = load_embedder({"CODE_SEARCH_OLLAMA_MODEL": "nomic-embed-text", "CODE_SEARCH_OLLAMA_URL": f.url})
    assert e.profile.model == "nomic-embed-text"


def test_cli_ollama_probe_reports_profile(ollama, tmp_path, monkeypatch):
    import io

    from workflows.apps.code_search.cli import main

    f = ollama(dim=5)
    monkeypatch.delenv("CODE_SEARCH_DEV_FAKE_EMBEDDER", raising=False)
    monkeypatch.setenv("CODE_SEARCH_OLLAMA_MODEL", "nomic-embed-text")
    monkeypatch.setenv("CODE_SEARCH_OLLAMA_URL", f.url)
    out = io.StringIO()
    assert main(["--home", str(tmp_path), "ollama-probe"], stdout=out, stderr=io.StringIO()) == 0
    text = out.getvalue()
    assert "digest sha256:abc123" in text and "dim 5" in text and "profile_id" in text

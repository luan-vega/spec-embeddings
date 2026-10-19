"""Embedder Ollama em loopback. Sem auto-pull, sem truncamento silencioso, sem vetores inválidos."""

from __future__ import annotations

import http.client
import ipaddress
import json
import math
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np

from ...domain.errors import CodeSearchError
from ...domain.model import EmbeddingProfile

DEFAULT_URL = "http://127.0.0.1:11434"
_BATCH = 16


def _unavailable(message: str) -> CodeSearchError:
    return CodeSearchError("embedder_unavailable", message)


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class OllamaClient:
    """HTTP mínimo (stdlib). Recusa qualquer host que não seja loopback."""

    def __init__(self, url: str = DEFAULT_URL, timeout_s: float = 60.0) -> None:
        parts = urlsplit(url)
        if parts.scheme != "http" or not _is_loopback(parts.hostname):
            raise _unavailable("the Ollama URL must be http on a loopback address")
        self._host = parts.hostname or "127.0.0.1"
        self._port = parts.port or 11434
        self._timeout = timeout_s

    def request(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> tuple[int, Any]:
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        conn = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)
        try:
            conn.request(method, path, body=payload, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
        except (OSError, http.client.HTTPException):
            raise _unavailable("Ollama is not reachable") from None
        finally:
            conn.close()
        try:
            return resp.status, json.loads(raw) if raw else None
        except ValueError:
            raise CodeSearchError("embedder_invalid_output", "Ollama returned invalid JSON") from None


def _model_matches(name: str, wanted: str) -> bool:
    return name == wanted or (":" not in wanted and name == f"{wanted}:latest")


def probe(client: OllamaClient, model: str) -> tuple[str, int, int]:
    """(digest, dim, max_input_tokens) do modelo já instalado. Nunca baixa nada."""
    status, tags = client.request("GET", "/api/tags")
    if status != 200 or not isinstance(tags, dict):
        raise _unavailable("Ollama did not list models")
    digest = next(
        (m.get("digest") for m in tags.get("models", []) if _model_matches(str(m.get("name")), model)), None
    )
    if not digest:
        raise _unavailable(f"model {model!r} is not installed in Ollama (pull it manually)")
    max_tokens = 0
    status, info = client.request("POST", "/api/show", {"model": model})
    if status == 200 and isinstance(info, dict):
        for key, value in (info.get("model_info") or {}).items():
            if key.endswith(".context_length") and isinstance(value, int):
                max_tokens = value
    vectors = _embed(client, model, ["probe"])
    return f"sha256:{digest}" if not str(digest).startswith("sha256:") else str(digest), vectors.shape[1], max_tokens


def _embed(client: OllamaClient, model: str, texts: Sequence[str]) -> np.ndarray:
    status, body = client.request("POST", "/api/embed", {"model": model, "input": list(texts), "truncate": False})
    if status == 400:
        raise CodeSearchError("embedder_invalid_output", "input rejected by the model (too long?)")
    if status != 200 or not isinstance(body, dict):
        raise _unavailable("Ollama failed to embed")
    try:
        arr = np.asarray(body["embeddings"], dtype=np.float32)
    except (KeyError, ValueError, TypeError):
        raise CodeSearchError("embedder_invalid_output", "malformed embeddings") from None
    if arr.ndim != 2 or arr.shape[0] != len(texts):
        raise CodeSearchError("embedder_invalid_output", "unexpected embeddings shape")
    return arr


class OllamaEmbedder:
    def __init__(
        self,
        model: str,
        *,
        url: str = DEFAULT_URL,
        doc_prefix: str = "",
        query_prefix: str = "",
        timeout_s: float = 60.0,
        client: OllamaClient | None = None,
    ) -> None:
        if not model:
            raise _unavailable("CODE_SEARCH_OLLAMA_MODEL is required")
        self._client = client or OllamaClient(url, timeout_s)
        self._model = model
        digest, dim, max_tokens = probe(self._client, model)
        self.profile = EmbeddingProfile(
            model=model, digest=digest, dim=dim, doc_prefix=doc_prefix, query_prefix=query_prefix,
            max_input_tokens=max_tokens,
        )

    def ready(self) -> bool:
        try:
            status, _ = self._client.request("GET", "/api/tags")
        except CodeSearchError:
            return False
        return status == 200

    def _checked(self, raw: np.ndarray) -> np.ndarray:
        if raw.shape[1] != self.profile.dim or not np.all(np.isfinite(raw)):
            raise CodeSearchError("embedder_invalid_output", "embedding has wrong dimension or non-finite values")
        norms = np.linalg.norm(raw, axis=1)
        if np.any(norms < 1e-8) or any(math.isnan(n) for n in norms):
            raise CodeSearchError("embedder_invalid_output", "zero-norm embedding")
        return (raw / norms[:, None]).astype(np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.profile.dim), dtype=np.float32)
        parts = []
        for i in range(0, len(texts), _BATCH):
            batch = [self.profile.doc_prefix + t for t in texts[i : i + _BATCH]]
            parts.append(self._checked(_embed(self._client, self._model, batch)))
        return np.concatenate(parts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._checked(_embed(self._client, self._model, [self.profile.query_prefix + text]))[0]

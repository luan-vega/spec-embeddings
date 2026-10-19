"""HTTP local em 127.0.0.1:8765 (http.server). Saúde responde mesmo sob carga: cada conexão tem sua thread
e /health nunca passa pelo executor de busca."""

from __future__ import annotations

import ipaddress
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ..core import CodeSearchError
from ..limits import OverloadedError
from ..obslog import log_event
from ..runtime import Runtime

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_BODY_BYTES = 64 * 1024
_ALLOWED_FIELDS = {"query", "mode", "repos", "languages", "path_prefix", "top_k"}


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def _str_list(value: Any, name: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise CodeSearchError("invalid_request", f"{name} must be a list of strings")
    return value


def parse_search_body(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise CodeSearchError("invalid_request", "body must be a JSON object")
    unknown = set(body) - _ALLOWED_FIELDS
    if unknown:
        raise CodeSearchError("invalid_request", "unknown field in request")
    query = body.get("query")
    if not isinstance(query, str):
        raise CodeSearchError("invalid_request", "query must be a string")
    path_prefix = body.get("path_prefix")
    if path_prefix is not None and not isinstance(path_prefix, str):
        raise CodeSearchError("invalid_request", "path_prefix must be a string")
    kwargs: dict[str, Any] = {
        "mode": body.get("mode", "hybrid"),
        "repos": _str_list(body.get("repos"), "repos"),
        "languages": _str_list(body.get("languages"), "languages"),
        "path_prefix": path_prefix,
        "top_k": body.get("top_k", 10),
    }
    return {"query": query, **kwargs}


class _Handler(BaseHTTPRequestHandler):
    server: "CodeSearchServer"
    server_version = "code-search"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:  # silencia o log padrão (pode conter a rota e o cliente)
        pass

    # ------------------------------------------------------------------ util
    def _send(self, status: int, payload: dict[str, Any], headers: dict[str, str] | None = None) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def _error(self, exc: CodeSearchError, request_id: str) -> None:
        headers = {"Retry-After": str(exc.retry_after)} if isinstance(exc, OverloadedError) else None
        self._send(
            exc.http_status,
            {"error": {"code": exc.code, "message": exc.message, "request_id": request_id}},
            headers,
        )

    def _host_ok(self) -> bool:
        """Barreira contra DNS rebinding: o Host precisa ser loopback."""
        host = (self.headers.get("Host") or "").strip()
        if host.startswith("["):
            name = host[1:].split("]")[0]
        else:
            name = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
        return _is_loopback(name)

    # ---------------------------------------------------------------- rotas
    def do_GET(self) -> None:
        rid = uuid.uuid4().hex
        if not self._host_ok():
            return self._error(CodeSearchError("invalid_request", "invalid Host header"), rid)
        if self.path == "/health":
            return self._send(200, {"status": "ok"})
        if self.path == "/ready":
            ok, checks = self.server.runtime.ready()
            return self._send(200 if ok else 503, {"ready": ok, "checks": checks})
        self._error(CodeSearchError("not_found", "route not found"), rid)

    def do_POST(self) -> None:
        rid = uuid.uuid4().hex
        started = time.perf_counter()
        trace: dict[str, Any] = {}
        try:
            if not self._host_ok():
                raise CodeSearchError("invalid_request", "invalid Host header")
            if self.path != "/v1/search":
                raise CodeSearchError("not_found", "route not found")
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":  # exige preflight de CORS: páginas web não conseguem postar aqui
                raise CodeSearchError("invalid_request", "Content-Type must be application/json")
            try:
                length = int(self.headers.get("Content-Length") or "")
            except ValueError:
                raise CodeSearchError("invalid_request", "Content-Length is required") from None
            if length < 0:
                raise CodeSearchError("invalid_request", "invalid Content-Length")
            if length > MAX_BODY_BYTES:
                raise CodeSearchError("payload_too_large", "request body is too large")
            try:
                body = json.loads(self.rfile.read(length))
            except ValueError:
                raise CodeSearchError("invalid_request", "body is not valid JSON") from None
            params = parse_search_body(body)
            service = self.server.runtime.search
            if service is None:
                raise CodeSearchError("embedder_unavailable", "search is not available")
            response = self.server.runtime.runner.run(
                service.search, params.pop("query"), request_id=rid, trace=trace, **params
            )
            self._send(200, response)
            self._log(rid, trace, started, response["status"])
        except CodeSearchError as exc:
            self._error(exc, rid)
            self._log(rid, trace, started, "error", code=exc.code)
        except Exception:  # noqa: BLE001 - nunca vaza detalhes internos
            self._error(CodeSearchError("internal_error", "internal error"), rid)
            self._log(rid, trace, started, "error", code="internal_error")

    def _method_not_allowed(self) -> None:
        self._error(CodeSearchError("method_not_allowed", "method not allowed"), uuid.uuid4().hex)

    do_PUT = do_DELETE = do_PATCH = _method_not_allowed

    def _log(self, rid: str, trace: dict[str, Any], started: float, status: str, code: str | None = None) -> None:
        log_event(
            "search",
            request_id=rid,
            status=status,
            code=code,
            mode=trace.get("mode"),
            profile_id=trace.get("profile_id"),
            generation=trace.get("generation"),
            duration_ms=(time.perf_counter() - started) * 1000,
            stages_ms=trace.get("stages_ms"),
            repos_searched=trace.get("repos_searched"),
            repos_unavailable=trace.get("repos_unavailable"),
            candidates=trace.get("candidates"),
            results=trace.get("results"),
        )


class CodeSearchServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, runtime: Runtime, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        if not _is_loopback(host):
            raise CodeSearchError("invalid_request", "the server only listens on loopback")
        super().__init__((host, port), _Handler)
        self.runtime = runtime


def serve(runtime: Runtime, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    server = CodeSearchServer(runtime, host, port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        runtime.runner.shutdown()

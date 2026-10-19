"""MCP por stdio (JSON-RPC 2.0, uma mensagem por linha). Somente leitura: não há ferramenta que escreva no índice."""

from __future__ import annotations

import json
from typing import Any, TextIO

from .api.server import parse_search_body
from .core import CodeSearchError
from .runtime import Runtime

SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
MAX_TEXT_CHARS = 4000  # por resultado devolvido ao cliente MCP

_TOOLS = [
    {
        "name": "search_code",
        "description": "Busca código Java nos repositórios indexados (híbrida, semântica ou lexical). Somente leitura.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "mode": {"type": "string", "enum": ["hybrid", "semantic", "lexical"]},
                "repos": {"type": "array", "items": {"type": "string"}},
                "languages": {"type": "array", "items": {"type": "string"}},
                "path_prefix": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "code_search_status",
        "description": "Estado do índice ativo e dos repositórios (revisão, data de indexação). Somente leitura.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "annotations": {"readOnlyHint": True},
    },
]


def _result(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _rpc_error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _tool_text(payload: dict[str, Any], is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}], "isError": is_error}


class McpServer:
    def __init__(self, runtime: Runtime) -> None:
        self._rt = runtime

    def handle(self, message: Any) -> dict[str, Any] | None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            return _rpc_error(message.get("id") if isinstance(message, dict) else None, -32600, "invalid request")
        method, msg_id = message["method"], message.get("id")
        is_notification = "id" not in message
        params = message.get("params") or {}
        if method == "initialize":
            wanted = params.get("protocolVersion") if isinstance(params, dict) else None
            version = wanted if wanted in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0]
            return _result(
                msg_id,
                {
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "code-search", "version": "0.1.0"},
                },
            )
        if method == "ping":
            return _result(msg_id, {})
        if method == "tools/list":
            return _result(msg_id, {"tools": _TOOLS})
        if method == "tools/call":
            return _result(msg_id, self._call(params)) if isinstance(params, dict) else _rpc_error(msg_id, -32602, "invalid params")
        if is_notification:
            return None  # notifications/initialized, notifications/cancelled...
        return _rpc_error(msg_id, -32601, "method not found")

    def _call(self, params: dict[str, Any]) -> dict[str, Any]:
        name, args = params.get("name"), params.get("arguments") or {}
        try:
            if name == "search_code":
                if self._rt.search is None:
                    raise CodeSearchError("embedder_unavailable", "search is not available")
                body = parse_search_body(args)
                response = self._rt.runner.run(self._rt.search.search, body.pop("query"), **body)
                for item in response["results"]:
                    item["text"] = item["text"][:MAX_TEXT_CHARS]
                return _tool_text(response)
            if name == "code_search_status":
                return _tool_text(self._rt.status())
            return _tool_text({"error": {"code": "invalid_request", "message": "unknown tool"}}, True)
        except CodeSearchError as exc:
            return _tool_text({"error": {"code": exc.code, "message": exc.message}}, True)


def serve_stdio(runtime: Runtime, stdin: TextIO, stdout: TextIO) -> None:
    server = McpServer(runtime)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            reply: dict[str, Any] | None = _rpc_error(None, -32700, "parse error")
        else:
            reply = server.handle(message)
        if reply is not None:
            stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            stdout.flush()

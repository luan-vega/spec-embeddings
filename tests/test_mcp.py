import io
import json

from workflows.apps.code_search.mcp_server import McpServer, serve_stdio

from rtutil import make_indexed_runtime


def call(server, method, params=None, msg_id=1):
    msg = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        msg["params"] = params
    return server.handle(msg)


def test_handshake_and_tool_list_is_read_only(tmp_path):
    server = McpServer(make_indexed_runtime(tmp_path))
    init = call(server, "initialize", {"protocolVersion": "2024-11-05"})
    assert init["result"]["protocolVersion"] == "2024-11-05" and init["result"]["serverInfo"]["name"] == "code-search"
    assert call(server, "initialize", {"protocolVersion": "1999-01-01"})["result"]["protocolVersion"] == "2025-06-18"
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert call(server, "ping")["result"] == {}
    tools = call(server, "tools/list")["result"]["tools"]
    assert [t["name"] for t in tools] == ["search_code", "code_search_status"]  # nenhuma ferramenta de escrita
    assert all(t["annotations"]["readOnlyHint"] for t in tools)


def test_search_code_and_status_tools(tmp_path):
    rt = make_indexed_runtime(tmp_path)
    server = McpServer(rt)
    res = call(server, "tools/call", {"name": "search_code", "arguments": {"query": "validar documento cpf", "top_k": 2}})["result"]
    assert res["isError"] is False
    payload = json.loads(res["content"][0]["text"])
    assert payload["status"] == "complete" and payload["results"][0]["symbol"] == "Cliente.validarCpf"

    status = call(server, "tools/call", {"name": "code_search_status", "arguments": {}})["result"]
    body = json.loads(status["content"][0]["text"])
    assert {r["repo_id"] for r in body["repos"]} == {"svc-pagamento", "svc-cliente"}
    assert all(r["indexed"] and r["chunks"] > 0 for r in body["repos"])
    assert str(tmp_path) not in status["content"][0]["text"]  # sem caminhos locais


def test_tool_errors_are_reported_inside_the_result(tmp_path):
    server = McpServer(make_indexed_runtime(tmp_path))
    bad = call(server, "tools/call", {"name": "search_code", "arguments": {"query": ""}})["result"]
    assert bad["isError"] is True and json.loads(bad["content"][0]["text"])["error"]["code"] == "invalid_request"
    none = call(server, "tools/call", {"name": "search_code", "arguments": {"query": "x", "repos": ["nada"]}})["result"]
    assert json.loads(none["content"][0]["text"])["error"]["code"] == "no_repos_available"
    unknown = call(server, "tools/call", {"name": "index_repo", "arguments": {}})["result"]
    assert unknown["isError"] is True


def test_protocol_errors(tmp_path):
    server = McpServer(make_indexed_runtime(tmp_path))
    assert call(server, "nao/existe")["error"]["code"] == -32601
    assert server.handle({"id": 1})["error"]["code"] == -32600
    assert server.handle("texto")["error"]["code"] == -32600


def test_stdio_loop(tmp_path):
    rt = make_indexed_runtime(tmp_path)
    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                "{nao json",
                "",
                json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
            ]
        )
        + "\n"
    )
    stdout = io.StringIO()
    serve_stdio(rt, stdin, stdout)
    replies = [json.loads(l) for l in stdout.getvalue().splitlines()]
    assert [r.get("id") for r in replies] == [1, None, 2]
    assert replies[1]["error"]["code"] == -32700

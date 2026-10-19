"""CLI: repo add|list|remove, policy init, ollama-probe, index, search, status, serve, mcp, eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

from .core import CodeSearchError
from .evaluation import check_thresholds, evaluate, load_dataset, load_thresholds
from .policy import write_default_policy
from .runtime import Runtime, build_runtime


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="code-search", description="Busca local de código Java multi-repo.")
    p.add_argument("--home", help="diretório de dados (padrão: ~/.workflows/code-search ou $CODE_SEARCH_HOME)")
    sub = p.add_subparsers(dest="command", required=True)

    repo = sub.add_parser("repo", help="gerencia o registro de repositórios")
    rsub = repo.add_subparsers(dest="repo_command", required=True)
    add = rsub.add_parser("add")
    add.add_argument("repo_id")
    add.add_argument("path")
    add.add_argument("--root", action="append", dest="roots", help="raiz relativa (padrão: src/main/java)")
    rsub.add_parser("list")
    rm = rsub.add_parser("remove")
    rm.add_argument("repo_id")

    pol = sub.add_parser("policy", help="política de exclusão")
    psub = pol.add_subparsers(dest="policy_command", required=True)
    init = psub.add_parser("init")
    init.add_argument("--force", action="store_true")

    sub.add_parser("ollama-probe", help="registra digest e dimensão do modelo e forma o perfil")

    idx = sub.add_parser("index", help="reindexa os repositórios habilitados")
    idx.add_argument("--repo", action="append", dest="repos")

    s = sub.add_parser("search", help="busca no índice ativo")
    s.add_argument("query")
    s.add_argument("--mode", default="hybrid", choices=["hybrid", "semantic", "lexical"])
    s.add_argument("--repo", action="append", dest="repos")
    s.add_argument("--language", action="append", dest="languages")
    s.add_argument("--path-prefix")
    s.add_argument("--top-k", type=int, default=10)
    s.add_argument("--json", action="store_true", dest="as_json")

    st = sub.add_parser("status", help="estado do índice e dos repositórios")
    st.add_argument("--json", action="store_true", dest="as_json")

    srv = sub.add_parser("serve", help="HTTP local em 127.0.0.1")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8765)

    sub.add_parser("mcp", help="servidor MCP por stdio (somente leitura)")

    ev = sub.add_parser("eval", help="gate de avaliação (UC1/UC2/UC3)")
    ev.add_argument("--dataset", required=True)
    ev.add_argument("--thresholds", required=True)
    ev.add_argument("--repo", action="append", dest="repos")
    ev.add_argument("--out")
    return p


def _print_search(response: dict[str, Any], out: TextIO) -> None:
    cov = response["coverage"]
    out.write(f"{response['status']} · gerações {response['index']['generation']} · repos: {', '.join(cov['searched'])}\n")
    for item in cov["unavailable"]:
        out.write(f"  indisponível: {item['repo']} ({item['reason']})\n")
    for i, r in enumerate(response["results"], start=1):
        out.write(
            f"{i:>2}. {r['repo']}:{r['source']}:{r['start_line']}-{r['end_line']}  {r['symbol']}  "
            f"(score {r['score']:.4f}, rev {r['repo_revision'][:8]}, {r['indexed_at']})\n"
        )


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_factory: Callable[..., Runtime] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    stdin: TextIO | None = None,
) -> int:
    out, err, inp = stdout or sys.stdout, stderr or sys.stderr, stdin or sys.stdin
    args = _parser().parse_args(argv)
    make = runtime_factory or build_runtime
    needs_embedder = args.command in {"index", "search", "serve", "mcp", "eval", "ollama-probe"}
    try:
        rt = make(args.home, with_embedder=needs_embedder)
        return _dispatch(args, rt, out, inp)
    except CodeSearchError as exc:
        err.write(f"erro [{exc.code}]: {exc.message}\n")
        return 1


def _dispatch(args: argparse.Namespace, rt: Runtime, out: TextIO, inp: TextIO) -> int:
    cmd = args.command
    if cmd == "repo":
        if args.repo_command == "add":
            entry = rt.registry.add(args.repo_id, args.path, args.roots or ("src/main/java",))
            out.write(f"repositório {entry.repo_id} registrado (roots: {', '.join(entry.roots)})\n")
        elif args.repo_command == "list":
            for e in rt.registry.list():
                out.write(f"{e.repo_id}\t{'habilitado' if e.enabled else 'desabilitado'}\t{', '.join(e.roots)}\n")
        else:
            removed = rt.registry.remove(args.repo_id)
            out.write("removido do registro (os snapshots não são apagados)\n" if removed else "repositório desconhecido\n")
            return 0 if removed else 1
        return 0
    if cmd == "policy":
        if rt.paths.policy.exists() and not args.force:
            raise CodeSearchError("invalid_request", "policy already exists; use --force to overwrite")
        write_default_policy(rt.paths.policy)
        out.write("política de exclusão padrão gravada\n")
        return 0
    if cmd == "ollama-probe":
        raise CodeSearchError("embedder_unavailable", "ollama-probe is not implemented yet (phase 4)")
    if cmd == "index":
        report = rt.indexer.index(args.repos)
        if report.status == "skipped":
            out.write(f"nada indexado: {report.reason}\n")
            return 1
        for r in report.repos:
            out.write(
                f"{r.repo_id}: {r.chunks} chunks, {r.embeddings_new} embeddings novos, {r.embeddings_cached} do cache, "
                f"{r.chunks_skipped_secret} descartados por segredo, rev {r.revision[:8]}\n"
            )
        out.write(f"geração {report.generation_id} ativa\n")
        return 0
    if cmd == "search":
        response = rt.runner.run(
            rt.search.search, args.query, mode=args.mode, repos=args.repos, languages=args.languages,
            path_prefix=args.path_prefix, top_k=args.top_k,
        )
        if args.as_json:
            out.write(json.dumps(response, ensure_ascii=False, indent=1) + "\n")
        else:
            _print_search(response, out)
        return 0
    if cmd == "status":
        status = rt.status()
        if args.as_json:
            out.write(json.dumps(status, ensure_ascii=False, indent=1) + "\n")
        else:
            idx = status["index"]
            out.write(f"política: {'válida' if status['policy']['valid'] else 'inválida (' + status['policy']['reason'] + ')'}\n")
            out.write(f"geração ativa: {idx['generation'] if idx and 'generation' in idx else 'nenhuma'}\n")
            for r in status["repos"]:
                out.write(f"{r['repo_id']}\t{'indexado' if r['indexed'] else 'sem índice'}\t{r['chunks']} chunks\t{r['repo_revision'] or '-'}\n")
        return 0
    if cmd == "serve":
        from .api.server import serve

        serve(rt, args.host, args.port)
        return 0
    if cmd == "mcp":
        from .mcp_server import serve_stdio

        serve_stdio(rt, inp, out)
        return 0
    if cmd == "eval":
        thresholds, digest = load_thresholds(args.thresholds)
        cases = load_dataset(args.dataset)
        report = evaluate(rt.search.search, cases, k=thresholds.get("k", 10), repos=args.repos)
        failures = check_thresholds(report, thresholds)
        report.update(thresholds_sha256=digest, passed=not failures, failures=failures)
        text = json.dumps(report, indent=1)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        out.write(text + "\n")
        return 0 if not failures else 1
    raise CodeSearchError("invalid_request", "unknown command")


if __name__ == "__main__":
    raise SystemExit(main())

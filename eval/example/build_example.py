"""Gera os repositórios Git de exemplo e o dataset rotulado UC1/UC2/UC3 a partir de ``corpus.py``.

Uso: python eval/example/build_example.py <diretório de saída>
Cria ``<saida>/repos/<repo>/`` (repositórios Git) e ``<saida>/dataset.jsonl``. Só biblioteca padrão.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import M, REPOS  # noqa: E402

_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "exemplo", "GIT_AUTHOR_EMAIL": "exemplo@example.com",
    "GIT_COMMITTER_NAME": "exemplo", "GIT_COMMITTER_EMAIL": "exemplo@example.com",
    "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
}


def java_path(repo: str, cls: str) -> str:
    return f"src/main/java/br/{REPOS[repo]}/{cls}.java"


def render_class(repo: str, cls: str, methods: list) -> str:
    out = [f"package br.{REPOS[repo]};", "", f"public class {cls} {{"]
    for i, (_, _, name, ret, params, intent, body) in enumerate(methods):
        out.append("")
        if i % 3 == 0:  # só um terço dos métodos tem javadoc, como em código real
            out.append(f"    /** {intent[0].upper() + intent[1:]}. */")
        out.append(f"    public {ret} {name}({params}) {{")
        out.extend("        " + line if j == 0 else line for j, line in enumerate(body.split("\n")))
        out.append("    }")
    out.append("}")
    return "\n".join(out) + "\n"


def code_to_code_query(ret: str, params: str, body: str) -> str:
    """UC3: o mesmo método com nomes trocados."""
    names = [p.split()[-1] for p in params.split(",")]
    renamed = {n: f"p{i}" for i, n in enumerate(names)}
    sig = params
    for old, new in renamed.items():
        sig = re.sub(rf"\b{old}\b", new, sig)
        body = re.sub(rf"\b{old}\b", new, body)
    flat = " ".join(line.strip() for line in body.split("\n"))
    return f"{ret} metodo({sig}) {{ {flat} }}"


def build(out: Path) -> Path:
    by_file: dict[tuple[str, str], list] = defaultdict(list)
    for m in M:
        by_file[(m[0], m[1])].append(m)
    for (repo, cls), methods in by_file.items():
        root = out / "repos" / repo
        target = root / java_path(repo, cls)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_class(repo, cls, methods), encoding="utf-8", newline="\n")
    for repo in REPOS:
        root = out / "repos" / repo
        if not (root / ".git").exists():
            subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True, env=_ENV)
        subprocess.run(["git", "-C", str(root), "add", "-A"], check=True, env=_ENV)
        subprocess.run(["git", "-C", str(root), "commit", "-q", "--allow-empty", "-m", "exemplo"], check=True, env=_ENV)
    dataset = out / "dataset.jsonl"
    dataset.write_text(dataset_text(), encoding="utf-8", newline="\n")
    return dataset


def dataset_text() -> str:
    lines = []
    for n, (repo, cls, name, ret, params, intent, body) in enumerate(M, start=1):
        relevant = [{"repo": repo, "source": java_path(repo, cls), "symbol": f"{cls}.{name}"}]
        for uc, query in (("UC1", intent), ("UC2", name), ("UC3", code_to_code_query(ret, params, body))):
            lines.append(json.dumps({"id": f"{uc.lower()}-{n:03d}", "uc": uc, "query": query, "relevant": relevant}, ensure_ascii=False))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    path = build(Path(sys.argv[1]))
    print(f"dataset: {path}")
    print(f"repositórios: {Path(sys.argv[1]) / 'repos'}")

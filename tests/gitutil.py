import os
import subprocess
from pathlib import Path

_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
    "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
}


def git(repo: Path, *args: str, stdin: bytes | None = None) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], input=stdin, capture_output=True, env=_ENV, check=True)
    return out.stdout.decode().strip()


def make_repo(path: Path, files: dict[str, str]) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True, env=_ENV)
    for rel, content in files.items():
        target = path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "init")
    return path


def commit_all(path: Path, message: str = "change") -> str:
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", message)
    return git(path, "rev-parse", "HEAD")


def add_symlink_entry(path: Path, rel: str, target: str) -> None:
    """Registra um symlink (modo 120000) no índice sem depender de suporte do SO."""
    sha = git(path, "hash-object", "-w", "--stdin", stdin=target.encode())
    git(path, "update-index", "--add", "--cacheinfo", f"120000,{sha},{rel}")

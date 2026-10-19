"""Registro de repositórios (repositories.json). Remover um repo altera só o registro."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ...domain.errors import CodeSearchError
from ...domain.model import DEFAULT_ROOTS, RepoEntry, normalize_root
from .file_store import _atomic_write_bytes
from .filelock import FileLock

_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
class Registry:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def _load(self) -> dict[str, RepoEntry]:
        try:
            data = json.loads(self.path.read_bytes())
            return {
                e["repo_id"]: RepoEntry(
                    repo_id=e["repo_id"],
                    path=e["path"],
                    roots=tuple(e["roots"]),
                    enabled=bool(e["enabled"]),
                    ref=e.get("ref", "HEAD"),
                )
                for e in data["repositories"]
            }
        except FileNotFoundError:
            return {}
        except (ValueError, KeyError, TypeError):
            raise CodeSearchError("invalid_request", "repositories.json is invalid") from None

    def _save(self, entries: dict[str, RepoEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {"version": 1, "repositories": [entries[k].to_dict() for k in sorted(entries)]}
        _atomic_write_bytes(self.path, json.dumps(body, indent=1).encode("utf-8"))

    def _lock(self) -> FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return FileLock(self.path.with_name(self.path.name + ".lock"), timeout=5)

    def list(self) -> list[RepoEntry]:
        entries = self._load()
        return [entries[k] for k in sorted(entries)]

    def get(self, repo_id: str) -> RepoEntry | None:
        return self._load().get(repo_id)

    def enabled(self) -> list[RepoEntry]:
        return [e for e in self.list() if e.enabled]

    def add(self, repo_id: str, path: Path | str, roots: tuple[str, ...] | list[str] = DEFAULT_ROOTS) -> RepoEntry:
        if not _REPO_ID_RE.match(repo_id):
            raise CodeSearchError("invalid_request", "invalid repo_id")
        repo_path = Path(path).resolve()
        if not repo_path.is_dir() or not (repo_path / ".git").exists():
            raise CodeSearchError("invalid_request", "path is not a Git repository")
        entry = RepoEntry(repo_id, str(repo_path), tuple(normalize_root(r) for r in roots) or DEFAULT_ROOTS)
        with self._lock():
            entries = self._load()
            if repo_id in entries:
                raise CodeSearchError("invalid_request", "repository already registered")
            entries[repo_id] = entry
            self._save(entries)
        return entry

    def remove(self, repo_id: str) -> bool:
        with self._lock():
            entries = self._load()
            if entries.pop(repo_id, None) is None:
                return False
            self._save(entries)
        return True

    def set_enabled(self, repo_id: str, enabled: bool) -> None:
        with self._lock():
            entries = self._load()
            if repo_id not in entries:
                raise CodeSearchError("invalid_request", "unknown repository")
            old = entries[repo_id]
            entries[repo_id] = RepoEntry(old.repo_id, old.path, old.roots, enabled, old.ref)
            self._save(entries)

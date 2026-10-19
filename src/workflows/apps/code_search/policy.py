"""Política de exclusão versionada + pré-filtro heurístico de segredos.

A política é validada antes de qualquer arquivo ser aberto. Ausente ou inválida significa
"indexa nada". O pertencimento de caminho usa componentes, nunca ``startswith``.
O filtro de segredos é uma barreira heurística, não um scanner dedicado.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .core import canonical_json
from .file_store import _atomic_write_bytes

DEFAULT_POLICY: dict[str, Any] = {
    "version": "1",
    "include_extensions": [".java"],
    "exclude_dirs": [
        ".git", ".svn", ".hg", ".idea", ".settings", ".gradle", ".mvn",
        "node_modules", "build", "target", "out", "bin", "dist", "generated-sources",
    ],
    "exclude_file_globs": [
        ".env*", "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore",
        "*.crt", "*.cer", "*.der", "id_rsa*", "id_ed25519*", "*.secret",
    ],
    "max_file_bytes": 1_000_000,
    "secret_patterns": [
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bgh[pousr]_[A-Za-z0-9]{30,}\b",
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.",
        r"(?i)\b(password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\b\s*[:=]\s*[\"'][^\"'\s]{6,}[\"']",
    ],
}
_KEYS = set(DEFAULT_POLICY)


def write_default_policy(path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(path, json.dumps(DEFAULT_POLICY, indent=1).encode("utf-8"))


@dataclass(frozen=True)
class ExclusionPolicy:
    valid: bool
    reason: str = ""
    version_id: str = "none"
    include_extensions: frozenset[str] = frozenset()
    exclude_dirs: frozenset[str] = frozenset()
    exclude_file_globs: tuple[str, ...] = ()
    max_file_bytes: int = 0
    secret_patterns: tuple[re.Pattern[str], ...] = field(default_factory=tuple)

    @classmethod
    def deny_all(cls, reason: str) -> "ExclusionPolicy":
        return cls(valid=False, reason=reason)

    @classmethod
    def load(cls, path: Path | str) -> "ExclusionPolicy":
        try:
            raw = Path(path).read_bytes()
        except FileNotFoundError:
            return cls.deny_all("policy_missing")
        except OSError:
            return cls.deny_all("policy_unreadable")
        try:
            return cls.from_dict(json.loads(raw))
        except ValueError:
            return cls.deny_all("policy_invalid")

    @classmethod
    def from_dict(cls, data: Any) -> "ExclusionPolicy":
        def invalid() -> "ExclusionPolicy":
            return cls.deny_all("policy_invalid")

        if not isinstance(data, dict) or set(data) != _KEYS:
            return invalid()
        version = data["version"]
        if not isinstance(version, str) or not version:
            return invalid()
        exts, dirs, globs, pats = (
            data["include_extensions"], data["exclude_dirs"], data["exclude_file_globs"], data["secret_patterns"],
        )
        for items in (exts, dirs, globs, pats):
            if not isinstance(items, list) or not all(isinstance(i, str) and i for i in items):
                return invalid()
        if not exts or not all(e.startswith(".") for e in exts):
            return invalid()
        if any("/" in d or "\\" in d for d in dirs):
            return invalid()
        size = data["max_file_bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            return invalid()
        try:
            compiled = tuple(re.compile(p) for p in pats)
        except re.error:
            return invalid()
        digest = hashlib.sha256(canonical_json(data)).hexdigest()[:8]
        return cls(
            valid=True,
            version_id=f"{version}-{digest}",
            include_extensions=frozenset(e.lower() for e in exts),
            exclude_dirs=frozenset(dirs),
            exclude_file_globs=tuple(g.lower() for g in globs),
            max_file_bytes=size,
            secret_patterns=compiled,
        )

    def allows(self, rel_path: str) -> bool:
        """Decide só pelo caminho relativo (posix), antes de abrir o arquivo."""
        if not self.valid:
            return False
        parts = PurePosixPath(rel_path.replace("\\", "/")).parts
        if not parts or parts[0] == "/" or ".." in parts:
            return False
        *dirs, name = parts
        if any(d in self.exclude_dirs for d in dirs):
            return False
        lowered = name.lower()
        if any(fnmatch.fnmatchcase(lowered, g) for g in self.exclude_file_globs):
            return False
        return PurePosixPath(lowered).suffix in self.include_extensions

    def contains_secret(self, text: str) -> bool:
        return any(p.search(text) for p in self.secret_patterns)

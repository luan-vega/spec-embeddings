"""Leitura dos arquivos rastreados no HEAD direto do Git (nunca do working tree)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from ...domain.errors import CodeSearchError
from ...domain.model import BlobEntry

_REGULAR_MODES = {"100644", "100755"}  # 120000 = symlink, 160000 = submódulo: ficam de fora
_BATCH = 200


def _unavailable(message: str = "repository is not readable") -> CodeSearchError:
    return CodeSearchError("repo_unavailable", message)


class GitSource:
    def __init__(self, repo_path: Path | str, ref: str = "HEAD", git: str = "git") -> None:
        self._repo = str(repo_path)
        self._ref = ref
        self._git = git
        self._env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}

    def _run(self, args: Sequence[str], stdin: bytes | None = None) -> bytes:
        try:
            proc = subprocess.run(
                [self._git, "-C", self._repo, *args],
                input=stdin,
                capture_output=True,
                env=self._env,
                check=False,
                timeout=300,
            )
        except (OSError, subprocess.SubprocessError):
            raise _unavailable("git is not available") from None
        if proc.returncode != 0:
            raise _unavailable()
        return proc.stdout

    def revision(self) -> str:
        return self._run(["rev-parse", "--verify", f"{self._ref}^{{commit}}"]).decode().strip()

    def list_blobs(self, roots: Iterable[str]) -> list[BlobEntry]:
        """Arquivos regulares rastreados sob as roots, na revisão resolvida."""
        rev = self.revision()
        out = self._run(["ls-tree", "-r", "-z", rev, "--", *roots])
        entries: list[BlobEntry] = []
        for record in out.split(b"\0"):
            if not record:
                continue
            meta, _, raw_path = record.partition(b"\t")
            mode, kind, sha = meta.decode().split(" ")
            if kind == "blob" and mode in _REGULAR_MODES:
                entries.append(BlobEntry(raw_path.decode("utf-8", errors="surrogateescape"), sha, mode))
        return sorted(entries, key=lambda e: e.path)

    def read_blobs(self, shas: Sequence[str]) -> Iterator[tuple[str, bytes]]:
        """(sha, conteúdo) em lotes; um único ``cat-file --batch`` por lote."""
        for i in range(0, len(shas), _BATCH):
            batch = shas[i : i + _BATCH]
            data = self._run(["cat-file", "--batch"], ("\n".join(batch) + "\n").encode())
            pos = 0
            for sha in batch:
                eol = data.index(b"\n", pos)
                header = data[pos:eol].split(b" ")
                if len(header) != 3 or header[1] != b"blob":
                    raise _unavailable("unexpected git object")
                size = int(header[2])
                start = eol + 1
                yield sha, data[start : start + size]
                pos = start + size + 1

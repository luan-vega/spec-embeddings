"""Lock de arquivo entre processos (e threads), portátil entre Windows e POSIX."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from ...domain.errors import CodeSearchError

if sys.platform == "win32":
    import msvcrt

    def _try_lock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


class FileLock:
    """Lock exclusivo para escritores. Leitores não usam lock."""

    def __init__(self, path: Path, timeout: float = 10.0, poll: float = 0.05) -> None:
        self._path = Path(path)
        self._timeout = timeout
        self._poll = poll
        self._fd: int | None = None

    def __enter__(self) -> "FileLock":
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        if os.fstat(fd).st_size == 0:
            os.write(fd, b"0")
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                _try_lock(fd)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise CodeSearchError("store_busy", "another writer holds the index lock") from None
                time.sleep(self._poll)
        self._fd = fd
        return self

    def __exit__(self, *exc: object) -> None:
        if self._fd is not None:
            try:
                _unlock(self._fd)
            finally:
                os.close(self._fd)
                self._fd = None

"""Concorrência limitada: nunca fila infinita. Excedeu o limite, responde 503 com retry-after."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Callable

from .core import CodeSearchError


class OverloadedError(CodeSearchError):
    def __init__(self, retry_after: int) -> None:
        super().__init__("overloaded", "the service is busy, retry later")
        self.retry_after = retry_after


class BoundedRunner:
    """Executor dedicado com limite de tarefas em execução e timeout por requisição.

    O timeout libera o cliente (504), mas a thread não é cancelável: a vaga só volta quando a tarefa
    termina de fato. Por isso o limite conta tarefas em execução, não requisições abertas.
    """

    def __init__(self, max_concurrent: int = 4, timeout_s: float = 30.0, retry_after: int = 1) -> None:
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._pool = ThreadPoolExecutor(max_workers=max_concurrent, thread_name_prefix="code-search")
        self._timeout = timeout_s
        self._retry_after = retry_after

    def run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if not self._slots.acquire(blocking=False):
            raise OverloadedError(self._retry_after)
        try:
            future: Future = self._pool.submit(fn, *args, **kwargs)
        except BaseException:
            self._slots.release()
            raise
        future.add_done_callback(lambda _f: self._slots.release())
        try:
            return future.result(timeout=self._timeout)
        except FutureTimeout:
            raise CodeSearchError("timeout", "the request took too long") from None

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

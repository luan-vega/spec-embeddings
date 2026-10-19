import io
import json
import logging
import threading
import time

import pytest

from workflows.apps.code_search import obslog
from workflows.apps.code_search.core import CodeSearchError
from workflows.apps.code_search.limits import BoundedRunner, OverloadedError


def test_runner_returns_value_and_propagates_errors():
    runner = BoundedRunner(max_concurrent=2, timeout_s=5)
    assert runner.run(lambda a, b=0: a + b, 1, b=2) == 3
    with pytest.raises(ZeroDivisionError):
        runner.run(lambda: 1 / 0)
    runner.shutdown()


def test_runner_rejects_instead_of_queueing_and_frees_slot_on_finish():
    runner = BoundedRunner(max_concurrent=1, timeout_s=5, retry_after=3)
    gate = threading.Event()
    started = threading.Event()

    def slow():
        started.set()
        gate.wait(5)
        return "ok"

    result = []
    t = threading.Thread(target=lambda: result.append(runner.run(slow)))
    t.start()
    assert started.wait(2)
    with pytest.raises(OverloadedError) as exc:
        runner.run(lambda: None)
    assert exc.value.code == "overloaded" and exc.value.http_status == 503 and exc.value.retry_after == 3
    gate.set()
    t.join(5)
    assert result == ["ok"]
    assert runner.run(lambda: "livre") == "livre"
    runner.shutdown()


def test_timeout_releases_client_but_slot_only_when_task_ends():
    runner = BoundedRunner(max_concurrent=1, timeout_s=0.1)
    gate = threading.Event()
    with pytest.raises(CodeSearchError) as exc:
        runner.run(lambda: gate.wait(5))
    assert exc.value.code == "timeout" and exc.value.http_status == 504
    with pytest.raises(OverloadedError):  # a tarefa ainda roda: a vaga continua ocupada
        runner.run(lambda: None)
    gate.set()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            assert runner.run(lambda: "ok") == "ok"
            break
        except OverloadedError:
            time.sleep(0.02)
    else:
        pytest.fail("slot never released")
    runner.shutdown()


def capture_logs():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(obslog._JsonFormatter())
    logger = obslog.get_logger()
    logger.addHandler(handler)
    return stream, lambda: logger.removeHandler(handler)


def test_log_values_are_whitelisted():
    stream, stop = capture_logs()
    try:
        obslog.log_event(
            "x", request_id="abc123", mode="hybrid", n=3, ratio=0.12345, ok=True, none=None,
            path="/home/luan/secret/repo/A.java", win="C:\\Users\\luan\\repo", query="como validar cpf do cliente",
            nested={"a": "ok", "b": "/etc/passwd"}, vec=[0.1, 0.2], obj=object(),
        )
    finally:
        stop()
    line = json.loads(stream.getvalue().strip())
    assert line["request_id"] == "abc123" and line["mode"] == "hybrid" and line["n"] == 3 and line["ratio"] == 0.123
    assert line["path"] == "<redacted>" and line["win"] == "<redacted>" and line["query"] == "<redacted>"
    assert line["nested"] == {"a": "ok", "b": "<redacted>"} and line["obj"] == "<redacted>"
    assert "/home/luan" not in stream.getvalue() and "validar cpf" not in stream.getvalue()

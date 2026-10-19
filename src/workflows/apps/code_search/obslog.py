"""Logs estruturados (uma linha JSON por evento) que não carregam código, vetores, credenciais,
caminhos absolutos nem o texto da consulta.

A garantia é do formato: só entram números, booleanos e strings curtas de um alfabeto restrito
(ids, nomes de modo e de status). Qualquer outra string vira ``<redacted>``.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from typing import Any

_SAFE_STR = re.compile(r"^[A-Za-z0-9_.:-]{0,64}$")
LOGGER_NAME = "code_search"


def _clean(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, str):
        return value if _SAFE_STR.match(value) else "<redacted>"
    if isinstance(value, dict):
        return {str(k)[:32]: _clean(v) for k, v in list(value.items())[:32]}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in list(value)[:32]]
    return "<redacted>"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(getattr(record, "fields", {"event": record.getMessage()}), sort_keys=True)


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    payload.update({k: _clean(v) for k, v in fields.items()})
    get_logger().info(event, extra={"fields": payload})

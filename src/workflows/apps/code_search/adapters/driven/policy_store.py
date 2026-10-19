"""Política de exclusão em JSON no disco. Ausente ou inválida significa "indexa nada"."""

from __future__ import annotations

import json
from pathlib import Path

from ...domain.policy import DEFAULT_POLICY, ExclusionPolicy
from .file_store import _atomic_write_bytes


def load_policy(path: Path | str) -> ExclusionPolicy:
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        return ExclusionPolicy.deny_all("policy_missing")
    except OSError:
        return ExclusionPolicy.deny_all("policy_unreadable")
    try:
        return ExclusionPolicy.from_dict(json.loads(raw))
    except ValueError:
        return ExclusionPolicy.deny_all("policy_invalid")


def write_default_policy(path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(path, json.dumps(DEFAULT_POLICY, indent=1).encode("utf-8"))

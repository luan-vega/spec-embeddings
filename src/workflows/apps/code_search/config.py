"""Caminhos padrão. Tudo local; nenhuma configuração aponta para fora de loopback."""

from __future__ import annotations

import os
from pathlib import Path


def default_home() -> Path:
    override = os.environ.get("CODE_SEARCH_HOME")
    return Path(override) if override else Path.home() / ".workflows" / "code-search"


class Paths:
    def __init__(self, home: Path | str | None = None) -> None:
        self.home = Path(home) if home else default_home()

    @property
    def registry(self) -> Path:
        return self.home / "repositories.json"

    @property
    def policy(self) -> Path:
        return self.home / "exclusion-policy.json"

    @property
    def cache(self) -> Path:
        return self.home / "cache"

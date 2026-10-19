"""Fakes determinísticos para testes de contrato. Nunca produzem vetores nulos."""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

from ...domain.model import EmbeddingProfile

_WORD = re.compile(r"[A-Za-z][a-z0-9]*|[A-Z]+(?![a-z])|\d+")


def simple_terms(text: str) -> dict[str, int]:
    """Tokenização mínima camelCase/snake_case, só para fakes de teste."""
    counts: dict[str, int] = {}
    for tok in re.split(r"[^A-Za-z0-9]+", text):
        for part in _WORD.findall(tok):
            term = part.lower()
            counts[term] = counts.get(term, 0) + 1
    return counts


class FakeEmbedder:
    def __init__(self, dim: int = 16, model: str = "fake-embed") -> None:
        self.profile = EmbeddingProfile(model=model, digest="sha256:fake", dim=dim)

    def _vec(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
        v = np.random.default_rng(seed).standard_normal(self.profile.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vec(self.profile.doc_prefix + t) for t in texts]) if texts else np.empty(
            (0, self.profile.dim), dtype=np.float32
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(self.profile.query_prefix + text)


class BagOfWordsEmbedder:
    """Embedder de teste: hashing trick sobre os termos. Textos com palavras em comum ficam próximos."""

    def __init__(self, dim: int = 64, model: str = "fake-bow") -> None:
        self.profile = EmbeddingProfile(model=model, digest="sha256:fake-bow", dim=dim)

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.profile.dim, dtype=np.float32)
        for term, count in simple_terms(text).items():
            h = int.from_bytes(hashlib.sha256(term.encode("utf-8")).digest()[:8], "little")
            v[h % self.profile.dim] += count
        if not v.any():
            v[0] = 1.0
        return v / np.linalg.norm(v)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.profile.dim), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)

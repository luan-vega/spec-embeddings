"""Encoder esparso determinístico para identificadores (camelCase e snake_case)."""

from __future__ import annotations

import re
from typing import Sequence

TOKENIZER_VERSION = "1"

_IDENT = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_PARTS = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def tokenize(text: str) -> list[str]:
    """Identificador inteiro (minúsculo) + suas partes. ``calcularJuros`` -> calcularjuros, calcular, juros."""
    terms: list[str] = []
    for ident in _IDENT.findall(text):
        parts = [p.lower() for p in _PARTS.findall(ident)]
        whole = ident.replace("_", "").replace("$", "").lower()
        if not whole:
            continue
        if len(parts) != 1 or parts[0] != whole:
            terms.append(whole)
        terms.extend(p for p in parts if len(p) > 1 or p.isdigit())
    return terms


class JavaSparseEncoder:
    version = TOKENIZER_VERSION

    def encode_documents(self, texts: Sequence[str]) -> list[dict[str, int]]:
        out: list[dict[str, int]] = []
        for text in texts:
            counts: dict[str, int] = {}
            for term in tokenize(text):
                counts[term] = counts.get(term, 0) + 1
            out.append(counts)
        return out

    def encode_query(self, text: str) -> dict[str, float]:
        return {term: 1.0 for term in tokenize(text)}

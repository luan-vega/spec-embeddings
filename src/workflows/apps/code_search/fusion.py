"""Fusão RRF e colapso de chunks sobrepostos."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

from .core import Hit

RRF_K = 60


def rrf(rankings: Iterable[Sequence[Hit]], k: int = RRF_K) -> list[Hit]:
    """Reciprocal Rank Fusion: usa só posições, então dispensa calibrar scores entre sinais."""
    fused: dict[tuple[str, int], float] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = (hit.repo_id, hit.row)
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    return [Hit(repo, row, score) for (repo, row), score in ordered]


def collapse_overlaps(
    hits: Sequence[Hit],
    locate: Callable[[Hit], tuple[str, int, int]],
    limit: int,
) -> list[Hit]:
    """Mantém o melhor de cada grupo de chunks sobrepostos (mesmo repo e arquivo), até ``limit``."""
    kept: list[Hit] = []
    spans: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for hit in hits:
        source, start, end = locate(hit)
        existing = spans.setdefault((hit.repo_id, source), [])
        if any(start <= e and s <= end for s, e in existing):
            continue
        existing.append((start, end))
        kept.append(hit)
        if len(kept) >= limit:
            break
    return kept

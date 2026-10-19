"""Harness de avaliação: Recall@k, MRR, nDCG@k e latência p50/p95, com limiares definidos antes da rodada.

Dataset (JSONL): {"id", "uc", "query", "relevant": [{"repo", "source", "symbol"?}]}
Limiares (JSON, versionado antes de rodar): ver ``eval/thresholds.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

MODES = ("hybrid", "semantic", "lexical")


@dataclass(frozen=True)
class Case:
    case_id: str
    uc: str
    query: str
    relevant: tuple[Mapping[str, str], ...]


def load_dataset(path: Path | str) -> list[Case]:
    cases = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            cases.append(Case(raw["id"], raw["uc"], raw["query"], tuple(raw["relevant"])))
        except (ValueError, KeyError, TypeError):
            raise ValueError(f"invalid dataset line {n}") from None
        if not cases[-1].relevant:
            raise ValueError(f"case {cases[-1].case_id} has no relevant items")
    return cases


def is_relevant(result: Mapping[str, Any], relevant: Sequence[Mapping[str, str]]) -> bool:
    for item in relevant:
        if (
            result["repo"] == item["repo"]
            and result["source"] == item["source"]
            and ("symbol" not in item or result["symbol"] == item["symbol"])
        ):
            return True
    return False


def recall_at_k(flags: Sequence[bool], total_relevant: int, k: int) -> float:
    return sum(flags[:k]) / total_relevant if total_relevant else 0.0


def reciprocal_rank(flags: Sequence[bool]) -> float:
    for i, hit in enumerate(flags, start=1):
        if hit:
            return 1.0 / i
    return 0.0


def ndcg_at_k(flags: Sequence[bool], total_relevant: int, k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, hit in enumerate(flags[:k]) if hit)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(total_relevant, k)))
    return dcg / ideal if ideal else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def load_thresholds(path: Path | str) -> tuple[dict[str, Any], str]:
    """Carrega os limiares e devolve também o hash do arquivo, registrado no relatório."""
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def evaluate(
    search: Callable[..., dict[str, Any]],
    cases: Sequence[Case],
    *,
    modes: Sequence[str] = MODES,
    k: int = 10,
    repos: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Roda cada caso em cada modo e agrega por UC e modo."""
    per_mode: dict[str, dict[str, dict[str, list[float]]]] = {}
    for mode in modes:
        for case in cases:
            started = time.perf_counter()
            response = search(case.query, mode=mode, top_k=k, repos=list(repos) if repos else None)
            elapsed = (time.perf_counter() - started) * 1000
            flags = [is_relevant(r, case.relevant) for r in response["results"]]
            total = len(case.relevant)
            bucket = per_mode.setdefault(mode, {}).setdefault(case.uc, {
                "recall@1": [], "recall@5": [], f"recall@{k}": [], "mrr": [], f"ndcg@{k}": [], "latency_ms": [],
            })
            bucket["recall@1"].append(recall_at_k(flags, total, 1))
            bucket["recall@5"].append(recall_at_k(flags, total, 5))
            bucket[f"recall@{k}"].append(recall_at_k(flags, total, k))
            bucket["mrr"].append(reciprocal_rank(flags))
            bucket[f"ndcg@{k}"].append(ndcg_at_k(flags, total, k))
            bucket["latency_ms"].append(elapsed)
    report: dict[str, Any] = {"k": k, "cases": len(cases), "modes": {}}
    for mode, by_uc in per_mode.items():
        report["modes"][mode] = {}
        for uc, metrics in by_uc.items():
            row = {name: sum(v) / len(v) for name, v in metrics.items() if name != "latency_ms"}
            row["n"] = len(metrics["mrr"])
            row["latency_p50_ms"] = percentile(metrics["latency_ms"], 50)
            row["latency_p95_ms"] = percentile(metrics["latency_ms"], 95)
            report["modes"][mode][uc] = row
    return report


def check_thresholds(report: Mapping[str, Any], thresholds: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Cada regra: {"uc", "mode", "metric", "min"|"max"|"min_delta_vs": "<modo base>"}. Devolve as falhas."""
    failures = []
    for rule in thresholds["rules"]:
        uc, mode, metric = rule["uc"], rule["mode"], rule["metric"]
        row = report["modes"].get(mode, {}).get(uc)
        if row is None or metric not in row:
            failures.append({**rule, "actual": None, "reason": "missing"})
            continue
        actual = row[metric]
        if "min" in rule and actual < rule["min"]:
            failures.append({**rule, "actual": actual})
        if "max" in rule and actual > rule["max"]:
            failures.append({**rule, "actual": actual})
        if "min_delta_vs" in rule:
            base = report["modes"].get(rule["min_delta_vs"], {}).get(uc, {}).get(metric)
            if base is None or actual - base < rule["delta"]:
                failures.append({**rule, "actual": actual, "baseline": base})
    return failures

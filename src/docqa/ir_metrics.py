"""Standard information-retrieval metrics (as reported by BEIR / MTEB) with bootstrap intervals.

All functions take, per query, the ranked list of retrieved ids and the set of relevant ids
(binary relevance, as in BEIR qrels with score 1).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Share of the relevant documents found in the top k."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """1 if at least one relevant document is in the top k (a.k.a. success@k)."""
    return float(any(doc in relevant for doc in ranked[:k]))


def mrr_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Reciprocal rank of the first relevant document within the top k."""
    for rank, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance (BEIR's main metric)."""
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, doc in enumerate(ranked[:k], 1) if doc in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


METRICS: dict[str, Callable[[Sequence[str], set[str], int], float]] = {
    "recall": recall_at_k,
    "hit": hit_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
}


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean (default 95%)."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    low = means[int((alpha / 2) * n_resamples)]
    high = means[min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)]
    return (low, high)


def evaluate_rankings(
    rankings: dict[str, list[str]],
    qrels: dict[str, set[str]],
    cutoffs: Sequence[tuple[str, int]] = (
        ("ndcg", 10),
        ("mrr", 10),
        ("recall", 1),
        ("recall", 5),
        ("recall", 10),
        ("recall", 100),
    ),
) -> dict[str, dict[str, float]]:
    """Mean and 95% bootstrap CI of each metric over the queries that have relevance labels."""
    queries = [q for q in qrels if qrels[q]]
    report: dict[str, dict[str, float]] = {}
    for name, k in cutoffs:
        per_query = [METRICS[name](rankings.get(q, []), qrels[q], k) for q in queries]
        mean = sum(per_query) / len(per_query) if per_query else 0.0
        low, high = bootstrap_ci(per_query)
        report[f"{name}@{k}"] = {
            "mean": round(mean, 4),
            "ci_low": round(low, 4),
            "ci_high": round(high, 4),
        }
    report["n_queries"] = {"mean": float(len(queries)), "ci_low": 0.0, "ci_high": 0.0}
    return report

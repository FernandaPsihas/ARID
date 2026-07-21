"""metrics.py -- retrieval-quality metrics for scoring a ranked chunk-id list
against a gold relevant-chunk-id set. Pure functions, no I/O, no dependency on
the rest of ARID -- reusable from bench_retrieval.py or a notebook alike.

All four assume binary relevance (a chunk either is or isn't in the gold set)
since that's what gold.jsonl currently encodes; nDCG still adds value over
plain recall because it rewards ranking the correct hit(s) higher, not just
including them somewhere in the top k.
"""

from __future__ import annotations

import math


def recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    """Of the gold-relevant chunks, what fraction appear anywhere in the top k?"""
    if not gold_ids:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & gold_ids) / len(gold_ids)


def precision_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    """Of the top k retrieved, what fraction are actually gold-relevant?"""
    top = ranked_ids[:k]
    if not top:
        return 0.0
    return len(set(top) & gold_ids) / len(top)


def mrr(ranked_ids: list[str], gold_ids: set[str]) -> float:
    """1 / (rank of the first relevant hit), 0 if none found. Not k-limited --
    the whole point is measuring how far down the full ranked list you'd have
    to look."""
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in gold_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    """Binary-relevance nDCG@k: rewards relevant hits ranked higher (1/log2(rank+1)
    discount), normalized against the best possible ordering for this query
    (all gold hits, up to k of them, at the very top)."""
    top = ranked_ids[:k]
    dcg = sum(1.0 / math.log2(i + 1) for i, cid in enumerate(top, start=1) if cid in gold_ids)
    ideal_hits = min(len(gold_ids), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def score_query(ranked_ids: list[str], gold_ids: set[str], ks: list[int]) -> dict:
    """All four metrics for one query, at every k in ks (mrr is k-independent)."""
    return {
        "recall": {k: recall_at_k(ranked_ids, gold_ids, k) for k in ks},
        "precision": {k: precision_at_k(ranked_ids, gold_ids, k) for k in ks},
        "ndcg": {k: ndcg_at_k(ranked_ids, gold_ids, k) for k in ks},
        "mrr": mrr(ranked_ids, gold_ids),
    }

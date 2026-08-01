"""metrics.py -- retrieval-quality metrics for scoring a ranked chunk-id list
against a gold relevant-chunk-id set. Pure functions, no I/O, no dependency on
the rest of ARID -- reusable from bench_retrieval.py or a notebook alike.

The first four assume binary relevance (a chunk either is or isn't in the gold
set) since that's what gold.jsonl currently encodes; nDCG still adds value
over plain recall because it rewards ranking the correct hit(s) higher, not
just including them somewhere in the top k.

Two more metrics below target specific pathologies the reranker/dedup work
cares about rather than generic ranking quality: constructor_surfacing_rate
(boilerplate ctors outranking the method with the real logic) and
multiplicity_coverage (queries whose gold answer is more than one chunk --
sibling implementations that must ALL be surfaced, not just one).
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


def is_constructor_chunk(chunk_id: str) -> bool:
    """True if chunk_id's fully-qualified symbol is a C++ constructor -- the
    identifier after the last '::' matches the class name immediately before
    it (e.g. dunereco::InfillChannels::InfillChannels: class InfillChannels,
    method also InfillChannels). Relies on make_chunk_id's
    f"{file}_{symbol}_{line}" convention: '::' only ever appears inside the
    symbol, never in a file path or the trailing line number, so splitting on
    '::' cleanly isolates the qualified name even though the file and the
    line number are glued on with plain underscores rather than '::'.
    Non-C++ chunks (no '::' at all, e.g. .fcl chunks) are never constructors."""
    parts = chunk_id.split("::")
    if len(parts) < 2:
        return False
    tail = parts[-1].rsplit("_", 1)
    method = tail[0] if len(tail) == 2 and tail[1].isdigit() else parts[-1]
    class_part = parts[-2]
    # Only the segment abutting the file path (len(parts) == 2, no namespace
    # level) is still glued to file text; deeper segments are clean identifiers.
    class_name = class_part.rsplit("_", 1)[-1] if len(parts) == 2 else class_part
    return bool(method) and method == class_name


def has_non_constructor_gold(gold_ids: set[str]) -> bool:
    """True if at least one gold-relevant chunk for this query is a real
    method rather than a constructor -- marks the queries where a constructor
    showing up in the ranking is unambiguously wrong, as opposed to a query
    (e.g. a ctor that reads its own config, like InfillChannels's does) whose
    correct answer genuinely IS the constructor. An aggregate
    constructor-surfacing rate should restrict to these queries, or a
    correctly-surfaced ctor gets averaged in as if it were noise."""
    return any(not is_constructor_chunk(cid) for cid in gold_ids)


def constructor_surfacing_rate(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    """Of the top k ranked results, what fraction are constructor chunks that
    are NOT themselves gold-relevant -- boilerplate ctor noise crowding out
    the method that actually contains the logic, which is the exact failure
    mode the reranker/dedup work was built to fix. A constructor that IS the
    gold answer never counts against this: only a constructor beating out the
    intended (non-constructor) answer counts as the pathology. Meaningful for
    any query, but most diagnostic when restricted to queries where
    has_non_constructor_gold is True."""
    top = ranked_ids[:k]
    if not top:
        return 0.0
    wrongly_surfaced = sum(1 for cid in top if is_constructor_chunk(cid) and cid not in gold_ids)
    return wrongly_surfaced / len(top)


def multiplicity_coverage(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    """For a multiplicity query -- more than one gold-relevant chunk, e.g. two
    sibling tool implementations of the same interface that must BOTH be
    surfaced -- what fraction of the DISTINCT gold ids appear anywhere in the
    top k. Arithmetically identical to recall_at_k; broken out under its own
    name so bench reports can isolate this exact subset (the RRF
    dedup-by-(file,symbol) work is specifically about not collapsing these
    down to a single hit) instead of averaging it in with every
    single-answer query, where the distinction is moot."""
    return recall_at_k(ranked_ids, gold_ids, k)


def multiplicity_coverage_summary(coverages: list[float]) -> dict[str, int]:
    """Classify a batch of multiplicity_coverage results (one per multiplicity
    query, all at the same k) into full / partial / zero coverage counts. An
    average alone would hide whether failures are all-or-nothing (a whole
    sibling implementation never surfaces) or death-by-a-thousand-cuts
    (every query gets half credit) -- those call for different fixes, which
    is why the team wants the breakdown, not just the mean."""
    full = sum(1 for c in coverages if c >= 1.0)
    zero = sum(1 for c in coverages if c <= 0.0)
    partial = len(coverages) - full - zero
    return {"full": full, "partial": partial, "zero": zero}


def score_query(ranked_ids: list[str], gold_ids: set[str], ks: list[int]) -> dict:
    """All per-query metrics for one query, at every k in ks (mrr is
    k-independent). constructor_surfacing is included unconditionally here
    (cheap, and harmless for queries where it doesn't apply); multiplicity
    metrics are deliberately NOT included since they only make sense for the
    subset of queries with more than one gold id -- callers should call
    multiplicity_coverage directly for that subset."""
    return {
        "recall": {k: recall_at_k(ranked_ids, gold_ids, k) for k in ks},
        "precision": {k: precision_at_k(ranked_ids, gold_ids, k) for k in ks},
        "ndcg": {k: ndcg_at_k(ranked_ids, gold_ids, k) for k in ks},
        "mrr": mrr(ranked_ids, gold_ids),
        "constructor_surfacing": {k: constructor_surfacing_rate(ranked_ids, gold_ids, k) for k in ks},
    }

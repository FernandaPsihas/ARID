"""Offline unit tests for the RRF/cross-encoder score blend in search.py's
_rerank() (see _blend_scores / _normalize in EGEpipeline/search.py).

IMPORTANT SCOPE NOTE: these tests check the BLENDING MATH ONLY, using
hand-picked fake scores standing in for a real rerank_server.py response. They
do NOT and CANNOT validate real retrieval quality -- that needs the actual
Qwen3-Reranker-0.6B model running on a GPU against the real Tier-1 index (see
EGEpipeline/eval/bench_ab.py / bench_retrieval.py), which this environment has
no access to. What's asserted here: (a) the blend collapses to the documented
pure-RRF / pure-reranker behaviour at the weight extremes, (b) normalization
handles degenerate/tied inputs without a divide-by-zero, (c) the formula is
monotonic in each input, and (d) on a small hand-built stand-in for the 8/1
CVNValidation-vs-CheckRecoEnergy lexical-trap bug, blending in the RRF score
keeps a well-retrieved candidate competitive instead of letting one bad
cross-encoder call fully bury it -- while also being honest that an extreme
enough cross-encoder outlier can still win (min-max normalization is relative
to the pool, so an outlier compresses everyone else's normalized score
downward too; the guarantee is "harder to fully bury", not "reranker can never
win").
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "EGEpipeline"))
from search import _blend_scores, _normalize


def test_normalize_basic_range():
    assert _normalize([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]


def test_normalize_tied_or_single_value_is_neutral_not_a_crash():
    # all-equal (or single-candidate) input is a ZeroDivisionError risk if not guarded
    assert _normalize([3.0, 3.0, 3.0]) == [0.5, 0.5, 0.5]
    assert _normalize([7.0]) == [0.5]


def test_blend_weight_1_reproduces_pure_reranker_order():
    # weight=1.0 is the pre-fix behaviour: cross-encoder score fully replaces RRF order.
    rrf = [0.030, 0.022, 0.018, 0.010]       # best RRF rank first (index 0)
    rerank = [0.62, 0.50, 0.30, 0.95]        # reranker likes candidate 3 best
    blended = _blend_scores(rrf, rerank, weight=1.0)
    order = sorted(range(4), key=lambda i: blended[i], reverse=True)
    assert order == [3, 0, 1, 2]  # exactly the rerank-score order


def test_blend_weight_0_reproduces_pure_rrf_order():
    rrf = [0.030, 0.022, 0.018, 0.010]
    rerank = [0.62, 0.50, 0.30, 0.95]
    blended = _blend_scores(rrf, rerank, weight=0.0)
    order = sorted(range(4), key=lambda i: blended[i], reverse=True)
    assert order == [0, 1, 2, 3]  # exactly the RRF order


def test_blend_softens_a_lexical_surface_match_burial():
    """Hand-built stand-in for the 8/1 bug: CheckRecoEnergy (index 0) is
    retrieved best (RRF rank 0) but the cross-encoder is fooled by
    CVNValidation's (index 3) lexical overlap with "validated" and scores it
    highest. Fake numbers, not a real model call -- this checks the blend
    formula's arithmetic, not reranker quality."""
    # index: 0=CheckRecoEnergy, 1=candidateB, 2=candidateC, 3=CVNValidation
    rrf_scores = [0.030, 0.022, 0.018, 0.010]     # CheckRecoEnergy retrieved best
    rerank_scores = [0.62, 0.50, 0.30, 0.95]      # reranker wrongly favors CVNValidation

    # Pure cross-encoder sort (the pre-fix behaviour) buries CheckRecoEnergy behind
    # the lexical false positive.
    pure_rerank_order = sorted(range(4), key=lambda i: rerank_scores[i], reverse=True)
    assert pure_rerank_order[0] == 3, pure_rerank_order  # CVNValidation wrongly wins

    # Blended at the module's default weight (ARID_RERANK_BLEND=0.65) lets
    # CheckRecoEnergy's strong retrieval signal back into the decision -- here
    # it's enough to overtake the lexical false positive.
    blended = _blend_scores(rrf_scores, rerank_scores, weight=0.65)
    blended_order = sorted(range(4), key=lambda i: blended[i], reverse=True)
    assert blended_order[0] == 0, blended_order  # CheckRecoEnergy wins once RRF has a vote

    # Not magic: min-max normalization is relative to the pool, so an extreme
    # enough cross-encoder outlier compresses everyone else's normalized score
    # toward 0 and can still win even after blending. The guarantee this change
    # provides is "harder to fully bury", not "reranker can never win" -- real
    # tuning of the weight has to happen against the live model, not this test.
    lopsided_rerank_scores = [0.62, 0.50, 0.30, 5.0]
    blended2 = _blend_scores(rrf_scores, lopsided_rerank_scores, weight=0.65)
    assert sorted(range(4), key=lambda i: blended2[i], reverse=True)[0] == 3


def test_blend_is_monotonic_in_the_reranker_score():
    """Raising one candidate's rerank score (holding RRF and everything else
    fixed) can only help, never hurt, its blended score -- a basic sanity
    check on the weighted-sum formula, independent of what a real model would
    actually score."""
    rrf = [0.02, 0.02, 0.02]
    low = _blend_scores(rrf, [0.1, 0.5, 0.9], weight=0.65)[0]
    high = _blend_scores(rrf, [0.8, 0.5, 0.9], weight=0.65)[0]
    assert high > low


def test_blend_is_monotonic_in_the_rrf_score():
    rerank = [0.02, 0.02, 0.02]
    low = _blend_scores([0.1, 0.5, 0.9], rerank, weight=0.65)[0]
    high = _blend_scores([0.8, 0.5, 0.9], rerank, weight=0.65)[0]
    assert high > low


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"{name} ok")
    print(f"all {len(tests)} rerank-blend tests passed")

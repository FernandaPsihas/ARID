"""search_codebase(): RRF fusion of dense + BM25, chunk-schema output (KAN-15/16).

Optional cross-encoder rerank stage: if ARID_RERANK_URL points at a running
rerank_server.py, the RRF-fused top-RERANK_POOL candidates get rescored by a
cross-encoder (reads query+chunk together, unlike the bi-encoder embedder),
then re-ordered before the final top_k cut by a weighted blend of that
cross-encoder score and the original RRF score (ARID_RERANK_BLEND, see
_blend_scores in _rerank) -- not a straight cross-encoder sort, so one bad
rerank call can't fully bury a candidate the retrieval side ranked well. Unset
the env var and it's a pure no-op -- behaviour is byte-identical to the old
RRF-only path, so existing callers are unaffected. If the URL is set but the
service is unreachable, a query degrades to RRF order with a warning rather
than failing (same philosophy as the dense-search fallback below).
"""

import json
import os
import sys
import urllib.request

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from search_bm25 import BM25, load_chunks

RRF_K = 60  # standard RRF constant; bigger = flatter rank weighting
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "..", "chunks.jsonl")
SCHEMA_FIELDS = ("id", "file", "start_line", "end_line", "symbol", "language", "text")

RERANK_URL = os.environ.get("ARID_RERANK_URL")          # e.g. http://localhost:8095; unset -> rerank off
RERANK_POOL = int(os.environ.get("ARID_RERANK_POOL", "40"))  # RRF candidates fed to the reranker
RERANK_SNIPPET = 2000  # chars of each chunk sent to the reranker (its own input is truncated too)
# 8/1: weight on the cross-encoder score in the final blend, in [0, 1]. 1.0 reproduces the
# old behaviour (reranker fully replaces RRF order); 0.0 ignores the reranker entirely. See
# _blend_scores() -- this is here so one bad reranker call (e.g. a lexical surface-match false
# positive like CVNValidation vs. "validated") can't fully bury a candidate that scored well
# on retrieval; the RRF side still has a say in the final order instead of being discarded.
RERANK_BLEND = float(os.environ.get("ARID_RERANK_BLEND", "0.65"))

_bm25_cache: dict[str, BM25 | None] = {}
def _get_bm25(chunks_path: str = CHUNKS_PATH) -> BM25 | None:
    """BM25 index over chunks_path, or None if it isn't present. Cached per
    path (not just one global singleton) so callers needing more than one
    corpus in the same process -- e.g. bench_ab.py's complete vs. partial
    index -- can hold both alive at once.

    On the shared-services setup a researcher can query purely against the shared
    Qdrant without ever extracting chunks.jsonl locally. Missing that file used to
    crash retrieval; now it just disables the BM25 half and we run dense-only.
    """
    if chunks_path not in _bm25_cache:  # ponytail: build once per path per process, cheap at ~6k chunks
        _bm25_cache[chunks_path] = None
        if not os.path.exists(chunks_path):
            # Fresh clone, no local chunks.jsonl: pull it back out of the shared
            # Qdrant so the BM25 half works without a manual sync_chunks step.
            # Runs at most once (the file then exists); degrades to dense-only if
            # Qdrant is unreachable, so it never blocks a query. Only applies to
            # the default chunks_path -- an explicit alternate path (e.g. a
            # partial-index corpus) is expected to already exist on disk.
            if chunks_path == CHUNKS_PATH:
                from sync_chunks import ensure_chunks
                ensure_chunks(chunks_path)
        try:
            _bm25_cache[chunks_path] = BM25(load_chunks(chunks_path))
        except FileNotFoundError:
            print(f"warning: {chunks_path} not found and could not be auto-synced from "
                  "Qdrant; BM25 disabled, using dense-only.", file=sys.stderr)
    return _bm25_cache[chunks_path]


def _rrf(*ranked_lists, k=RRF_K):
    """Reciprocal rank fusion. Inputs are ranked result dicts keyed by chunk_id."""
    scores, meta = {}, {}
    for results in ranked_lists:
        for rank, r in enumerate(results):
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            meta[cid] = r
    return scores, meta


def _normalize(xs: list[float]) -> list[float]:
    """Min-max squash a list of scores to [0, 1]. A tied/degenerate list (all
    equal, incl. a single element) maps to a constant 0.5 -- neutral, so a flat
    signal doesn't get to arbitrarily out-vote the other one in _blend_scores."""
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return [0.5] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]


def _blend_scores(rrf_scores: list[float], rerank_scores: list[float],
                  weight: float = RERANK_BLEND) -> list[float]:
    """Combine RRF scores and cross-encoder scores into one final ranking score
    per candidate, so the cross-encoder REORDERS the pool rather than fully
    REPLACING the retrieval-side ordering. Each list is min-max normalized to
    [0, 1] within this candidate pool before blending, because the two scores
    live on unrelated, incomparable scales: RRF scores are sums of reciprocal
    ranks (tiny positive floats, no fixed range), while reranker scores are
    either P(yes) in [0, 1] (causal, the default) or an unbounded classification
    logit (seqcls) -- blending raw values would let whichever side happens to
    have larger magnitudes silently dominate regardless of `weight`.

    `weight` is the trust placed in the cross-encoder: 1.0 reproduces the old
    "reranker fully replaces RRF order" behaviour, 0.0 ignores the reranker
    entirely. Pure function of two equal-length float lists -- no model, no I/O
    -- so it's covered by an offline unit test with fake scores standing in for
    a real reranker call.
    """
    r_n = _normalize(rrf_scores)
    x_n = _normalize(rerank_scores)
    return [weight * x + (1.0 - weight) * r for x, r in zip(x_n, r_n)]


def _rerank(query: str, candidates: list[dict]) -> list[dict]:
    """Reorder candidates by a blend of the cross-encoder rerank score and the
    original RRF score, if a rerank service is configured and reachable.
    Stdlib-only (urllib) so the query venv needs no torch/transformers -- the
    model lives in rerank_server.py inside the GPU container. Any failure (no
    URL, service down, bad response) returns the input unchanged, so a query
    never breaks on rerank being unavailable.

    Blended rather than a straight cross-encoder sort (see _blend_scores):
    a single bad cross-encoder call -- e.g. a lexical surface-match false
    positive like scoring CVNValidation::analyze above CheckRecoEnergy::analyze
    for a "validated" query, purely because "Validation" echoes "validated" --
    can shove a well-retrieved candidate out of the running entirely under a
    pure cross-encoder sort. Blending means the RRF side still has a vote in
    the final order instead of being fully discarded."""
    if not RERANK_URL or not candidates:
        return candidates
    try:
        payload = json.dumps({"query": query,
                              "passages": [c["text"][:RERANK_SNIPPET] for c in candidates]}).encode("utf-8")
        req = urllib.request.Request(RERANK_URL.rstrip("/") + "/rerank", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            scores = json.loads(resp.read())["scores"]
        if len(scores) != len(candidates):  # defensive: mismatched response, don't trust it
            raise ValueError(f"got {len(scores)} scores for {len(candidates)} candidates")
        rrf_scores = [c["score"] for c in candidates]  # RRF order-derived score, set in search_codebase
        blended = _blend_scores(rrf_scores, scores)
        order = sorted(range(len(candidates)), key=lambda i: blended[i], reverse=True)
        return [candidates[i] for i in order]
    except Exception as e:  # rerank is an enhancement, never a hard dependency
        print(f"warning: rerank unavailable ({e}); using RRF order", file=sys.stderr)
        return candidates


def search_codebase(query: str, top_k: int = 10, pool: int = 10,
                     chunks_path: str = CHUNKS_PATH, collection: str | None = None) -> list[dict]:
    """Hybrid search -> list of chunk-schema dicts (+ rrf score), best first.

    Deduped to one chunk per (file, symbol) group so a cluster of near-duplicate
    top hits for one implementation can't crowd a second, distinct one out of top_k
    (7/20 meeting item 2 -- "return all implementations, not just one").

    chunks_path/collection override the BM25 corpus / dense collection for
    callers that need to query something other than the default live
    setup -- e.g. bench_ab.py running the same hybrid search against a
    separate partial-index corpus and Qdrant collection, without touching
    the production alias.
    """
    # With rerank on, pull a bigger candidate pool from each leg so the reranker
    # actually has the buried-but-correct chunks to float up (the pixel-map
    # CreateMap methods sat at RRF ranks 15/20 -- a pool of 10 never sees them).
    retrieve_pool = max(pool, RERANK_POOL) if RERANK_URL else pool
    try:
        from embed_store import search_dense  # lazy: pulls in qdrant/ollama only when used
        dense = search_dense(query, top_k=retrieve_pool, collection=collection)
    except Exception as e:  # ponytail: qdrant/ollama down (dead tunnel) -> BM25-only, not a crash
        print(f"warning: dense search unavailable ({e}); falling back to BM25-only", file=sys.stderr)
        dense = []
    bm25 = _get_bm25(chunks_path)
    sparse = bm25.search(query, top_k=retrieve_pool) if bm25 is not None else []
    if not dense and not sparse:
        # both halves are unavailable (no shared store reachable AND no local
        # chunks.jsonl) -- nothing to retrieve. Callers treat [] as "no hits".
        print("warning: neither dense (Qdrant) nor BM25 (chunks.jsonl) retrieval is "
              "available; returning no results", file=sys.stderr)
        return []
    scores, meta = _rrf(dense, sparse)

    # Candidate dicts in RRF order. Rerank (if configured) reorders them by a
    # blend of the cross-encoder score and this RRF score BEFORE dedup, so it
    # sees every sibling/overload and can pick the one that actually answers
    # the query -- then dedup keeps the best per (file, symbol) preserving that
    # order, then top_k (7/20 item 2 + 7/24 rerank, 8/1 blend).
    ordered_ids = sorted(scores, key=scores.get, reverse=True)
    if RERANK_URL:
        ordered_ids = ordered_ids[:RERANK_POOL]
    candidates = []
    for cid in ordered_ids:
        r = meta[cid]
        d = {f: r[f] for f in SCHEMA_FIELDS if f != "id"}
        d["id"] = r["chunk_id"]
        d["score"] = scores[cid]
        candidates.append(d)
    candidates = _rerank(query, candidates)

    out, seen = [], set()
    for d in candidates:
        group = (d["file"], d["symbol"])
        if group in seen:
            continue
        seen.add(group)
        out.append(d)
        if len(out) >= top_k:
            break
    return out


def _selfcheck():
    dense = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "x"}]
    bm = [{"chunk_id": "b"}, {"chunk_id": "c"}]
    scores, _ = _rrf(dense, bm)
    assert max(scores, key=scores.get) == "b", scores  # only one in both lists
    assert scores["a"] > scores["x"], scores           # rank matters within a list
    print("ok")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "--test":
        _selfcheck()
        sys.exit(0)
    query = " ".join(sys.argv[1:])
    for r in search_codebase(query):
        print(f"  [{r['score']:.4f}]  {r['file']}  L{r['start_line']}-{r['end_line']}  {r['symbol']}")

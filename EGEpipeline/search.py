"""search_codebase(): RRF fusion of dense + BM25, chunk-schema output (KAN-15/16)."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from search_bm25 import BM25, load_chunks

RRF_K = 60  # standard RRF constant; bigger = flatter rank weighting
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "..", "chunks.jsonl")
SCHEMA_FIELDS = ("id", "file", "start_line", "end_line", "symbol", "language", "text")

_bm25 = None
_bm25_tried = False
def _get_bm25() -> BM25 | None:
    """BM25 index over the local chunks.jsonl, or None if it isn't present.

    On the shared-services setup a researcher can query purely against the shared
    Qdrant without ever extracting chunks.jsonl locally. Missing that file used to
    crash retrieval; now it just disables the BM25 half and we run dense-only.
    """
    global _bm25, _bm25_tried
    if _bm25 is None and not _bm25_tried:  # ponytail: build once per process, cheap at ~6k chunks
        _bm25_tried = True
        if not os.path.exists(CHUNKS_PATH):
            # Fresh clone, no local chunks.jsonl: pull it back out of the shared
            # Qdrant so the BM25 half works without a manual sync_chunks step.
            # Runs at most once (the file then exists); degrades to dense-only if
            # Qdrant is unreachable, so it never blocks a query.
            from sync_chunks import ensure_chunks
            ensure_chunks(CHUNKS_PATH)
        try:
            _bm25 = BM25(load_chunks(CHUNKS_PATH))
        except FileNotFoundError:
            print(f"warning: {CHUNKS_PATH} not found and could not be auto-synced from "
                  "Qdrant; BM25 disabled, using dense-only.", file=sys.stderr)
    return _bm25


def _rrf(*ranked_lists, k=RRF_K):
    """Reciprocal rank fusion. Inputs are ranked result dicts keyed by chunk_id."""
    scores, meta = {}, {}
    for results in ranked_lists:
        for rank, r in enumerate(results):
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            meta[cid] = r
    return scores, meta


def search_codebase(query: str, top_k: int = 10, pool: int = 10) -> list[dict]:
    """Hybrid search -> list of chunk-schema dicts (+ rrf score), best first.

    Deduped to one chunk per (file, symbol) group so a cluster of near-duplicate
    top hits for one implementation can't crowd a second, distinct one out of top_k
    (7/20 meeting item 2 -- "return all implementations, not just one").
    """
    try:
        from embed_store import search_dense  # lazy: pulls in qdrant/ollama only when used
        dense = search_dense(query, top_k=pool)
    except Exception as e:  # ponytail: qdrant/ollama down (dead tunnel) -> BM25-only, not a crash
        print(f"warning: dense search unavailable ({e}); falling back to BM25-only", file=sys.stderr)
        dense = []
    bm25 = _get_bm25()
    sparse = bm25.search(query, top_k=pool) if bm25 is not None else []
    if not dense and not sparse:
        # both halves are unavailable (no shared store reachable AND no local
        # chunks.jsonl) -- nothing to retrieve. Callers treat [] as "no hits".
        print("warning: neither dense (Qdrant) nor BM25 (chunks.jsonl) retrieval is "
              "available; returning no results", file=sys.stderr)
        return []
    scores, meta = _rrf(dense, sparse)
    best_per_group = {}
    for cid in scores:
        group = (meta[cid]["file"], meta[cid]["symbol"])
        if group not in best_per_group or scores[cid] > scores[best_per_group[group]]:
            best_per_group[group] = cid
    ranked = sorted(best_per_group.values(), key=scores.get, reverse=True)[:top_k]
    out = []
    for cid in ranked:
        r = meta[cid]
        d = {f: r[f] for f in SCHEMA_FIELDS if f != "id"}
        d["id"] = r["chunk_id"]
        d["score"] = scores[cid]
        out.append(d)
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

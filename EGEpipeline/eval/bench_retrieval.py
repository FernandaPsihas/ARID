"""bench_retrieval.py -- scores the real hybrid retrieval (search.py's
search_codebase, dense+BM25 RRF fusion) against gold.jsonl using recall@k,
precision@k, nDCG@k, and MRR.

Different question from bench_ab.py/AB_TESTING.md: that one checks whether the
pipeline hallucinates when context is missing (BM25-only, complete vs partial
index). This one checks how good the actual production retrieval is at
finding the right chunk at all, ranked properly -- against the real dense+BM25
hybrid, no index-swapping.

Hard-fails if dense retrieval isn't reachable (dead Qdrant/Ollama tunnel)
instead of letting search_codebase() silently degrade to BM25-only -- a run
scored while degraded would look like a real hybrid-search number and isn't
one. If you WANT a BM25-only baseline on purpose, that's a separate,
explicit thing to add later, not a silent fallback here.

Usage:
    python bench_retrieval.py                    # full run against gold.jsonl
    python bench_retrieval.py --ks 1 3 5 10       # (default) which k values to report
    python bench_retrieval.py --limit 3           # smoke test on the first 3 queries
    python bench_retrieval.py --gold path/to.jsonl --pool 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EGE_ROOT = os.path.dirname(HERE)
DEFAULT_GOLD = os.path.join(HERE, "gold.jsonl")

sys.path.insert(0, EGE_ROOT)
from metrics import score_query  # noqa: E402


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def preflight_dense_check() -> None:
    """Hard-fail here, loudly, rather than let search_codebase() catch this
    and quietly fall back to BM25-only -- see module docstring."""
    from embed_store import search_dense
    try:
        search_dense("preflight connectivity check", top_k=1)
    except Exception as e:
        sys.exit(
            "FATAL: dense retrieval (Qdrant/Ollama) is not reachable -- "
            f"{type(e).__name__}: {e}\n"
            "This harness requires the real hybrid search, not a silent "
            "BM25-only degrade. Check the tunnel (see memory: 'ARID Remote "
            "Models Access') and retry."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--pool", type=int, default=None, help="top_k passed to search_codebase (default: max(ks))")
    ap.add_argument("--limit", type=int, default=None, help="only score the first N gold queries")
    ap.add_argument("--embed-model", default=None,
                     help="override ARID_EMBED_MODEL (which Ollama model embeds the query) -- "
                          "must match whatever model built --collection's vectors")
    ap.add_argument("--collection", default=None,
                     help="override ARID_EMBED_COLLECTION (which Qdrant collection to search)")
    args = ap.parse_args()

    pool = args.pool or max(args.ks)

    # Must be set before embed_store.py (imported transitively via search.py) reads
    # them at module level -- that's why these env vars are set here, before any
    # ARID import, rather than passed as function args deeper in the call chain.
    if args.embed_model:
        os.environ["ARID_EMBED_MODEL"] = args.embed_model
    if args.collection:
        os.environ["ARID_EMBED_COLLECTION"] = args.collection

    preflight_dense_check()
    from search import search_codebase  # after preflight, and after sys.path is set up

    gold_rows = load_jsonl(args.gold)
    if args.limit:
        gold_rows = gold_rows[: args.limit]

    per_query = []
    for row in gold_rows:
        question = row["question"]
        gold_ids = set(row["relevant_chunk_ids"])
        results = search_codebase(question, top_k=pool)
        ranked_ids = [r["id"] for r in results]
        scores = score_query(ranked_ids, gold_ids, args.ks)
        per_query.append({
            "question": question,
            "gold_ids": sorted(gold_ids),
            "ranked_ids": ranked_ids,
            "scores": scores,
        })
        print(f"scored: {question[:70]!r}  mrr={scores['mrr']:.2f}  "
              f"recall@{args.ks[-1]}={scores['recall'][args.ks[-1]]:.2f}")

    n = len(per_query) or 1
    agg = {
        "recall": {k: sum(q["scores"]["recall"][k] for q in per_query) / n for k in args.ks},
        "precision": {k: sum(q["scores"]["precision"][k] for q in per_query) / n for k in args.ks},
        "ndcg": {k: sum(q["scores"]["ndcg"][k] for q in per_query) / n for k in args.ks},
        "mrr": sum(q["scores"]["mrr"] for q in per_query) / n,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(HERE, f"retrieval_report_{ts}.json")
    md_path = os.path.join(HERE, f"retrieval_report_{ts}.md")

    embed_model = os.environ.get("ARID_EMBED_MODEL", "qwen3-embedding:0.6b")
    collection = os.environ.get("ARID_EMBED_COLLECTION", "dunereco")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"gold": args.gold, "ks": args.ks, "pool": pool,
                    "embed_model": embed_model, "collection": collection,
                    "aggregate": agg, "per_query": per_query}, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# ARID retrieval benchmark ({ts})\n\n")
        f.write(f"Embed model: `{embed_model}`  |  Qdrant collection: `{collection}`\n\n")
        f.write(f"Gold set: `{args.gold}` ({n} queries)  |  pool (search top_k): {pool}\n\n")
        f.write("## Aggregate\n\n")
        f.write("| metric | " + " | ".join(f"@{k}" for k in args.ks) + " |\n")
        f.write("|---" * (len(args.ks) + 1) + "|\n")
        for name in ("recall", "precision", "ndcg"):
            f.write(f"| {name} | " + " | ".join(f"{agg[name][k]:.2f}" for k in args.ks) + " |\n")
        f.write(f"\n**MRR**: {agg['mrr']:.3f}\n\n")
        f.write("## Per-query\n\n")
        for q in per_query:
            f.write(f"### {q['question']}\n\n")
            f.write(f"- gold: {q['gold_ids']}\n")
            f.write(f"- retrieved (ranked): {q['ranked_ids']}\n")
            f.write(f"- mrr={q['scores']['mrr']:.2f}, "
                     f"recall@{args.ks[-1]}={q['scores']['recall'][args.ks[-1]]:.2f}, "
                     f"precision@{args.ks[-1]}={q['scores']['precision'][args.ks[-1]]:.2f}, "
                     f"ndcg@{args.ks[-1]}={q['scores']['ndcg'][args.ks[-1]]:.2f}\n\n")

    print(f"\nwrote {md_path}\nwrote {json_path}")
    print("aggregate:", json.dumps(agg, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

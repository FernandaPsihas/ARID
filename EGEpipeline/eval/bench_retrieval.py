"""bench_retrieval.py -- scores the real hybrid retrieval (search.py's
search_codebase, dense+BM25 RRF fusion) against gold.jsonl using recall@k,
precision@k, nDCG@k, and MRR -- plus two pathology-specific metrics:
constructor-surfacing rate (boilerplate ctors outranking the real-logic
method) and multiplicity coverage (queries whose gold answer is more than one
chunk, all of which must surface).

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
from metrics import (  # noqa: E402
    constructor_surfacing_rate,
    has_non_constructor_gold,
    multiplicity_coverage,
    multiplicity_coverage_summary,
    score_query,
)


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
        is_multiplicity = len(gold_ids) > 1
        ctor_applicable = has_non_constructor_gold(gold_ids)
        multiplicity_scores = (
            {k: multiplicity_coverage(ranked_ids, gold_ids, k) for k in args.ks}
            if is_multiplicity else None
        )
        per_query.append({
            "question": question,
            "gold_ids": sorted(gold_ids),
            "ranked_ids": ranked_ids,
            "scores": scores,
            "is_multiplicity": is_multiplicity,
            "ctor_pathology_applicable": ctor_applicable,
            "multiplicity_scores": multiplicity_scores,
        })
        note = ""
        if is_multiplicity:
            note += f"  multiplicity_cov@{args.ks[-1]}={multiplicity_scores[args.ks[-1]]:.2f}"
        if ctor_applicable:
            # top-1 specifically -- computed directly rather than via scores[...][1]
            # since 1 isn't guaranteed to be in args.ks (e.g. --ks 3 5 10).
            note += f"  ctor_surfacing@1={constructor_surfacing_rate(ranked_ids, gold_ids, 1):.2f}"
        print(f"scored: {question[:70]!r}  mrr={scores['mrr']:.2f}  "
              f"recall@{args.ks[-1]}={scores['recall'][args.ks[-1]]:.2f}{note}")

    n = len(per_query) or 1
    agg = {
        "recall": {k: sum(q["scores"]["recall"][k] for q in per_query) / n for k in args.ks},
        "precision": {k: sum(q["scores"]["precision"][k] for q in per_query) / n for k in args.ks},
        "ndcg": {k: sum(q["scores"]["ndcg"][k] for q in per_query) / n for k in args.ks},
        "mrr": sum(q["scores"]["mrr"] for q in per_query) / n,
        "constructor_surfacing": {k: sum(q["scores"]["constructor_surfacing"][k] for q in per_query) / n
                                   for k in args.ks},
    }

    # Constructor-surfacing rate, restricted to queries where a constructor
    # showing up would unambiguously be wrong (see has_non_constructor_gold) --
    # the unrestricted number above is diluted by queries (like InfillChannels'
    # ctor) whose correct answer genuinely is a constructor.
    ctor_rows = [q for q in per_query if q["ctor_pathology_applicable"]]
    n_ctor = len(ctor_rows) or 1
    agg["constructor_surfacing_applicable"] = {
        "n_queries": len(ctor_rows),
        "rate": {k: sum(q["scores"]["constructor_surfacing"][k] for q in ctor_rows) / n_ctor for k in args.ks},
    }

    # Multiplicity coverage: subset of queries with >1 gold id, i.e. sibling
    # implementations that must ALL surface (see gold.jsonl notes on rows 17-20).
    multi_rows = [q for q in per_query if q["is_multiplicity"]]
    n_multi = len(multi_rows) or 1
    agg["multiplicity"] = {
        "n_queries": len(multi_rows),
        "mean_coverage": {k: sum(q["multiplicity_scores"][k] for q in multi_rows) / n_multi for k in args.ks},
        "breakdown": {k: multiplicity_coverage_summary([q["multiplicity_scores"][k] for q in multi_rows])
                      for k in args.ks},
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
        for name in ("recall", "precision", "ndcg", "constructor_surfacing"):
            f.write(f"| {name} | " + " | ".join(f"{agg[name][k]:.2f}" for k in args.ks) + " |\n")
        f.write(f"\n**MRR**: {agg['mrr']:.3f}\n\n")

        f.write("## Constructor-vs-method surfacing\n\n")
        f.write(f"Restricted to the {agg['constructor_surfacing_applicable']['n_queries']} "
                f"queries (of {n}) where gold contains a real method, so a constructor "
                "showing up would unambiguously be crowding out the right answer:\n\n")
        f.write("| @k | " + " | ".join(str(k) for k in args.ks) + " |\n")
        f.write("|---" * (len(args.ks) + 1) + "|\n")
        f.write("| rate | " + " | ".join(
            f"{agg['constructor_surfacing_applicable']['rate'][k]:.2f}" for k in args.ks) + " |\n\n")

        f.write("## Multiplicity coverage\n\n")
        f.write(f"{agg['multiplicity']['n_queries']} of {n} queries have more than one gold id "
                "(sibling implementations that must all surface).\n\n")
        f.write("| @k | mean coverage | full | partial | zero |\n")
        f.write("|---|---|---|---|---|\n")
        for k in args.ks:
            b = agg["multiplicity"]["breakdown"][k]
            f.write(f"| {k} | {agg['multiplicity']['mean_coverage'][k]:.2f} | "
                    f"{b['full']} | {b['partial']} | {b['zero']} |\n")

        f.write("\n## Per-query\n\n")
        for q in per_query:
            f.write(f"### {q['question']}\n\n")
            f.write(f"- gold: {q['gold_ids']}\n")
            f.write(f"- retrieved (ranked): {q['ranked_ids']}\n")
            f.write(f"- mrr={q['scores']['mrr']:.2f}, "
                     f"recall@{args.ks[-1]}={q['scores']['recall'][args.ks[-1]]:.2f}, "
                     f"precision@{args.ks[-1]}={q['scores']['precision'][args.ks[-1]]:.2f}, "
                     f"ndcg@{args.ks[-1]}={q['scores']['ndcg'][args.ks[-1]]:.2f}\n")
            if q["ctor_pathology_applicable"]:
                f.write(f"- constructor_surfacing@{args.ks[-1]}="
                        f"{q['scores']['constructor_surfacing'][args.ks[-1]]:.2f} "
                        "(gold contains a real method -- a constructor here is wrong)\n")
            if q["is_multiplicity"]:
                f.write(f"- multiplicity_coverage@{args.ks[-1]}="
                        f"{q['multiplicity_scores'][args.ks[-1]]:.2f} "
                        f"({len(q['gold_ids'])} distinct gold ids)\n")
            f.write("\n")

    print(f"\nwrote {md_path}\nwrote {json_path}")
    print("aggregate:", json.dumps(agg, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

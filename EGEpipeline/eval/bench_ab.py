"""bench_ab.py -- A/B robustness test: does the pipeline correctly decline (or
at least go quiet) on a module that's been deliberately removed from the
index, instead of hallucinating an answer for it? Different question from
bench_retrieval.py: that one scores how good retrieval is at finding the
right chunk at all. This one runs the REAL hybrid search (dense + BM25 + RRF,
same search_codebase() path production uses) over two corpora -- the full
chunks.jsonl ("complete") and a copy with a few modules' chunks stripped out
("partial", its own separate Qdrant collection, never the live alias) -- and
checks whether removing a module actually changes retrieval/generation
behavior the way it should.

Rebuilt 2026-07-23: the original script (7/12-7/13) was lost with no git
history, only a stale .pyc survived. Reconstructed from its own past output
(the ab_report_*.json files it wrote, still on disk) rather than guessed --
those pin down the exact schema (excluded_modules/elapsed_s/generate/counts/
rows, and per-row question/note/gold_modules/excluded/complete_hit/
partial_hit/expect_partial_hit/verdict/complete/partial). The original ran
BM25-only; upgraded 2026-07-24 to the real hybrid path since that's what's
actually deployed -- testing the weaker BM25-only path was a lower bar than
what real researchers get. The partial side needs its own Qdrant collection
(built once via `--rebuild-partial-collection`, embedded under whatever
ARID_EMBED_MODEL is current), kept fully separate from the live `dunereco`
alias so this never touches production data.

ONE DELIBERATE CHANGE from the original verdict logic: the old heuristic used
a text-pattern match on the generated answer to guess "did it decline or
hallucinate", and the 7/13 slide deck says that heuristic had a false-positive
bug (flagged an honest decline as `hallucination_suspected`). Rather than
reproduce a heuristic known to have been wrong, this version checks something
concrete instead: does the partial-index answer cite a file/symbol that
belongs to an EXCLUDED module gold chunk that was never in its own retrieved
context? If so, that's not "sounds hallucinate-y", it's "cited something it
was never given" -- a much harder signal to get a false positive on. See
`_cites_ungrounded_gold` below (itself patched once already -- see the
"::"-qualified-symbol-only comment there for a real false positive this
caught and fixed).

Usage:
    python bench_ab.py                                # hybrid retrieval, no generation
    python bench_ab.py --generate                     # also runs answer.py's _generate()
    python bench_ab.py --limit 3                       # smoke test
    python bench_ab.py --rebuild-partial               # regenerate chunks_partial.jsonl + manifest
    python bench_ab.py --rebuild-partial-collection    # (re)embed the partial corpus into its own
                                                        # Qdrant collection -- run this once after any
                                                        # --rebuild-partial, or after the live embed
                                                        # model changes, before trusting dense results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EGE_ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(EGE_ROOT)
DEFAULT_GOLD = os.path.join(HERE, "gold.jsonl")
DEFAULT_COMPLETE = os.path.join(REPO_ROOT, "chunks.jsonl")
DEFAULT_PARTIAL = os.path.join(REPO_ROOT, "chunks_partial.jsonl")
DEFAULT_EXCLUDED = ["CVN", "FDSelections", "HitFinderDUNE"]  # matches the manifest already on disk
DEFAULT_PARTIAL_COLLECTION = "dunereco_partial_nomiccode"  # separate from the live "dunereco" alias

sys.path.insert(0, EGE_ROOT)
from search_bm25 import load_chunks  # noqa: E402
from search import search_codebase  # noqa: E402


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_partial_chunks(complete_path: str, partial_path: str, excluded_modules: list[str]) -> None:
    """Drop every chunk from a dunereco/<Module>/... path where <Module> is
    excluded, write the survivors to partial_path, and write a sibling
    .manifest.json recording what got dropped and why -- same schema as the
    manifest already on disk from the original 7/12 run, so old and new
    partial indexes stay comparable."""
    chunks = load_chunks(complete_path)
    dropped_by_module = {m: 0 for m in excluded_modules}
    dropped_files = set()
    kept = []
    for c in chunks:
        parts = c["file"].split("/")
        module = parts[1] if len(parts) > 1 and parts[0] == "dunereco" else None
        if module in dropped_by_module:
            dropped_by_module[module] += 1
            dropped_files.add(c["file"])
            continue
        kept.append(c)

    with open(partial_path, "w", encoding="utf-8") as f:
        for c in kept:
            f.write(json.dumps(c) + "\n")

    manifest = {
        "in_path": os.path.relpath(complete_path, REPO_ROOT),
        "out_path": os.path.relpath(partial_path, REPO_ROOT),
        "excluded_modules": excluded_modules,
        "kept_chunks": len(kept),
        "dropped_chunks": len(chunks) - len(kept),
        "dropped_by_module": dropped_by_module,
        "dropped_files": sorted(dropped_files),
    }
    manifest_path = partial_path.rsplit(".jsonl", 1)[0] + ".manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {partial_path} ({len(kept)} kept, {len(chunks) - len(kept)} dropped) "
          f"and {manifest_path}")


def rebuild_partial_collection(partial_chunks_path: str, collection: str) -> None:
    """(Re)embed the partial corpus into its own standalone Qdrant collection
    -- never the live "dunereco" alias, so this can't affect production
    queries. Uses whatever ARID_EMBED_MODEL is currently the default (the
    same model backing the live alias), so complete vs. partial stays a fair
    apples-to-apples comparison."""
    import embed_store as es
    import qdrant_index as qi

    chunks = load_chunks(partial_chunks_path)
    dim = es._dim()
    if es._client.collection_exists(collection):
        es._client.delete_collection(collection)
    qi.create_collection(es._client, collection, dim)
    es.index_into(collection, chunks)
    print(f"embedded {len(chunks)} chunks into standalone collection {collection!r} "
          f"under {es.EMBED_MODEL}")


def _gold_modules(relevant_chunk_ids: list[str], file_by_id: dict[str, str]) -> set[str]:
    mods = set()
    for cid in relevant_chunk_ids:
        file = file_by_id.get(cid)
        if not file:
            continue
        parts = file.split("/")
        if len(parts) > 1 and parts[0] == "dunereco":
            mods.add(parts[1])
    return mods


def _cites_ungrounded_gold(answer_text: str, gold_ids: list[str], retrieved_ids: set[str],
                            file_by_id: dict[str, str]) -> bool:
    """True if the answer confidently names a file/symbol belonging to a gold
    chunk that was never actually in its own retrieved context -- i.e. it
    couldn't have gotten this from what it was given, so it's citing from
    training-data memorization rather than the provided snippets. Checked
    per gold chunk id that's NOT in retrieved_ids (the ones it should have no
    way to know about).

    Only trusts specific signals: the file's basename, or a qualified symbol
    name (contains "::"). A bare unqualified symbol (e.g. FHiCL top-level
    block names like "services"/"physics"/"source") is too generic -- those
    recur across nearly every .fcl file in the corpus, so matching on one
    alone is a false-hallucination-flag waiting to happen (caught this for
    real: the FHiCL services-config gold query flagged a hallucination purely
    because the answer legitimately discussed *other*, actually-retrieved
    .fcl files' "services" blocks)."""
    for cid in gold_ids:
        if cid in retrieved_ids:
            continue  # it legitimately had this one, not a hallucination signal
        file = file_by_id.get(cid, "")
        symbol = cid[len(file) + 1:].rsplit("_", 1)[0] if file and cid.startswith(file + "_") else ""
        basename = file.rsplit("/", 1)[-1] if file else ""
        if basename and basename in answer_text:
            return True
        if symbol and "::" in symbol and symbol in answer_text:
            return True
    return False


def score_row(question: str, gold_ids: list[str], gold_modules: set[str], note: str,
              excluded_modules: list[str], complete_chunks_path: str, partial_chunks_path: str,
              partial_collection: str, file_by_id: dict[str, str], top_k: int, generate: bool) -> dict:
    excluded = bool(gold_modules & set(excluded_modules))

    # complete side: real hybrid search against the live alias (collection=None -> default),
    # exactly the path a real query takes. partial side: same hybrid search, but pointed at
    # the standalone partial corpus + its own separate collection.
    complete_hits = search_codebase(question, top_k=top_k, chunks_path=complete_chunks_path)
    partial_hits = search_codebase(question, top_k=top_k, chunks_path=partial_chunks_path,
                                    collection=partial_collection)
    complete_ids = {h["id"] for h in complete_hits}
    partial_ids = {h["id"] for h in partial_hits}
    complete_hit = any(cid in complete_ids for cid in gold_ids)
    partial_hit = any(cid in partial_ids for cid in gold_ids)

    complete_chunks = [{"file": h["file"], "start_line": h["start_line"], "end_line": h["end_line"],
                         "symbol": h["symbol"], "language": h["language"], "text": h["text"]}
                        for h in complete_hits]
    partial_chunks = [{"file": h["file"], "start_line": h["start_line"], "end_line": h["end_line"],
                        "symbol": h["symbol"], "language": h["language"], "text": h["text"]}
                       for h in partial_hits]

    complete_answer = partial_answer = None
    if generate:
        from answer import _generate  # lazy: keeps retrieval-only mode dependency-free
        complete_answer = _generate(question, complete_chunks) if complete_chunks else "(no chunks retrieved)"
        partial_answer = _generate(question, partial_chunks) if partial_chunks else "(no chunks retrieved)"

    if not complete_hit:
        verdict = "complete_retrieval_miss"
    elif excluded:
        if partial_hit:
            verdict = "leak"  # excluded module's chunk still surfaced in the "partial" index -- exclusion is broken
        elif generate and _cites_ungrounded_gold(partial_answer, gold_ids, partial_ids, file_by_id):
            verdict = "hallucination_suspected"
        else:
            verdict = "declined_correctly"
    else:
        verdict = "ok" if partial_hit else "partial_retrieval_miss"

    row = {
        "question": question,
        "note": note,
        "gold_modules": sorted(gold_modules),
        "excluded": excluded,
        "complete_hit": complete_hit,
        "partial_hit": partial_hit,
        "expect_partial_hit": not excluded,
        "verdict": verdict,
        "complete": {"query": question, "chunks_path": complete_chunks_path,
                     "retrieved": [{"id": h["id"], "file": h["file"], "symbol": h["symbol"],
                                    "score": h["score"]} for h in complete_hits],
                     "answer": complete_answer},
        "partial": {"query": question, "chunks_path": partial_chunks_path, "collection": partial_collection,
                    "retrieved": [{"id": h["id"], "file": h["file"], "symbol": h["symbol"],
                                   "score": h["score"]} for h in partial_hits],
                    "answer": partial_answer},
    }
    return row


def write_reports(rows: list[dict], excluded_modules: list[str], elapsed_s: float, generate: bool,
                   out_dir: str, complete_chunks_path: str, partial_chunks_path: str,
                   partial_collection: str) -> tuple[str, str]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    json_path = os.path.join(out_dir, f"ab_report_{ts}.json")
    md_path = os.path.join(out_dir, f"ab_report_{ts}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"excluded_modules": excluded_modules, "elapsed_s": elapsed_s,
                    "generate": generate, "counts": counts, "rows": rows}, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# ARID A/B report ({ts})\n\n")
        f.write(f"Complete: `{complete_chunks_path}` (live alias)  \n")
        f.write(f"Partial: `{partial_chunks_path}` + collection `{partial_collection}` "
                f"(excluded: {', '.join(excluded_modules)})\n\n")
        f.write(f"Verdict counts: {json.dumps(counts)}\n\n")
        for r in rows:
            f.write(f"## {r['question']}\n\n")
            f.write(f"- gold modules: {r['gold_modules']}  |  excluded: {r['excluded']}  |  "
                    f"verdict: **{r['verdict']}**\n")
            f.write(f"- complete hit: {r['complete_hit']}  |  partial hit: {r['partial_hit']}\n\n")
            if generate:
                f.write(f"**Complete answer:**\n\n{r['complete']['answer']}\n\n")
                f.write(f"**Partial answer:**\n\n{r['partial']['answer']}\n\n")
            f.write("\n---\n\n")

    print(f"\nwrote {md_path}\nwrote {json_path}")
    print("verdict counts:", json.dumps(counts, indent=2))
    return json_path, md_path


def _preflight_dense_check(chunks_path: str, collection: str | None) -> None:
    """Hard-fail loudly here rather than let search_codebase() catch this and
    quietly degrade to BM25-only -- a run scored while degraded is a weaker
    test than the one this is supposed to be, and would look identical in
    the report unless we check up front (same reasoning as
    bench_retrieval.py's preflight check)."""
    from embed_store import search_dense
    try:
        search_dense("preflight connectivity check", top_k=1, collection=collection)
    except Exception as e:
        sys.exit(
            f"FATAL: dense retrieval against collection={collection or '(live alias)'} is not "
            f"reachable -- {type(e).__name__}: {e}\n"
            "This harness is meant to test the real hybrid path, not a silent BM25-only degrade. "
            "If the partial collection doesn't exist yet, run --rebuild-partial-collection first."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--complete-chunks", default=DEFAULT_COMPLETE)
    ap.add_argument("--partial-chunks", default=DEFAULT_PARTIAL)
    ap.add_argument("--partial-collection", default=DEFAULT_PARTIAL_COLLECTION,
                     help="standalone Qdrant collection for the partial corpus -- never the live alias")
    ap.add_argument("--excluded-modules", nargs="+", default=DEFAULT_EXCLUDED)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="only score the first N gold queries")
    ap.add_argument("--generate", action="store_true",
                     help="also run answer.py's _generate() on each side and check for citations "
                          "the model couldn't have gotten from its own retrieved context")
    ap.add_argument("--rebuild-partial", action="store_true",
                     help="regenerate --partial-chunks (+ its .manifest.json) from --complete-chunks "
                          "before scoring, instead of using whatever's already on disk")
    ap.add_argument("--rebuild-partial-collection", action="store_true",
                     help="(re)embed --partial-chunks into --partial-collection before scoring -- "
                          "needed once after any --rebuild-partial, or after the live embed model changes")
    args = ap.parse_args()

    if args.rebuild_partial or not os.path.exists(args.partial_chunks):
        build_partial_chunks(args.complete_chunks, args.partial_chunks, args.excluded_modules)

    if args.rebuild_partial_collection:
        rebuild_partial_collection(args.partial_chunks, args.partial_collection)

    _preflight_dense_check(args.complete_chunks, None)
    _preflight_dense_check(args.partial_chunks, args.partial_collection)

    t0 = time.time()
    complete_chunks_data = load_chunks(args.complete_chunks)
    file_by_id = {c["id"]: c["file"] for c in complete_chunks_data}

    gold_rows = load_jsonl(args.gold)
    if args.limit:
        gold_rows = gold_rows[: args.limit]

    rows = []
    for row in gold_rows:
        gold_ids = row["relevant_chunk_ids"]
        mods = _gold_modules(gold_ids, file_by_id)
        r = score_row(row["question"], gold_ids, mods, row.get("note", ""), args.excluded_modules,
                      args.complete_chunks, args.partial_chunks, args.partial_collection,
                      file_by_id, args.top_k, args.generate)
        rows.append(r)
        print(f"scored: {row['question'][:70]!r}  verdict={r['verdict']}")

    elapsed_s = time.time() - t0
    write_reports(rows, args.excluded_modules, elapsed_s, args.generate, HERE,
                  args.complete_chunks, args.partial_chunks, args.partial_collection)
    return 0


if __name__ == "__main__":
    sys.exit(main())

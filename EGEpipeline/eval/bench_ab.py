"""bench_ab.py -- A/B robustness test: does retrieval correctly decline (or at
least go quiet) on a module that's been deliberately removed from the index,
instead of hallucinating an answer for it? Different question from
bench_retrieval.py: that one scores how good the real hybrid retrieval is at
finding the right chunk at all. This one takes BM25-only retrieval (stdlib,
no Qdrant/Ollama needed for the retrieval half) over two corpora -- the full
chunks.jsonl ("complete") and a copy with a few modules' chunks stripped out
("partial") -- and checks whether removing a module actually changes
retrieval/generation behavior the way it should.

Rebuilt 2026-07-23: the original script (7/12-7/13) was lost with no git
history, only a stale .pyc survived. Reconstructed from its own past output
(the ab_report_*.json files it wrote, still on disk) rather than guessed --
those pin down the exact schema (excluded_modules/elapsed_s/generate/counts/
rows, and per-row question/note/gold_modules/excluded/complete_hit/
partial_hit/expect_partial_hit/verdict/complete/partial) and confirmed the
retrieval side is BM25-only (the scores in the old reports are BM25
magnitudes, not RRF-fused ones).

ONE DELIBERATE CHANGE from the original: the old verdict logic used a
text-pattern heuristic on the generated answer to guess "did it decline or
hallucinate", and the 7/13 slide deck says that heuristic had a false-positive
bug (flagged an honest decline as `hallucination_suspected`). Rather than
reproduce a heuristic known to have been wrong, this version checks something
concrete instead: does the partial-index answer cite a file/symbol that
belongs to an EXCLUDED module gold chunk that was never in its own retrieved
context? If so, that's not "sounds hallucinate-y", it's "cited something it
was never given" -- a much harder signal to get a false positive on. See
`_cites_ungrounded_gold` below.

Usage:
    python bench_ab.py                              # BM25-only, no generation
    python bench_ab.py --generate                   # also runs answer.py's _generate()
    python bench_ab.py --limit 3                     # smoke test
    python bench_ab.py --rebuild-partial             # regenerate chunks_partial.jsonl + manifest first
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

sys.path.insert(0, EGE_ROOT)
from search_bm25 import BM25, load_chunks  # noqa: E402


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
              excluded_modules: list[str], complete_bm25: BM25, partial_bm25: BM25,
              file_by_id: dict[str, str], top_k: int, generate: bool) -> dict:
    excluded = bool(gold_modules & set(excluded_modules))

    complete_hits = complete_bm25.search(question, top_k=top_k)
    partial_hits = partial_bm25.search(question, top_k=top_k)
    complete_ids = {h["chunk_id"] for h in complete_hits}
    partial_ids = {h["chunk_id"] for h in partial_hits}
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
        "complete": {"query": question, "chunks_path": DEFAULT_COMPLETE,
                     "retrieved": [{"id": h["chunk_id"], "file": h["file"], "symbol": h["symbol"],
                                    "score": h["score"]} for h in complete_hits],
                     "answer": complete_answer},
        "partial": {"query": question, "chunks_path": DEFAULT_PARTIAL,
                    "retrieved": [{"id": h["chunk_id"], "file": h["file"], "symbol": h["symbol"],
                                   "score": h["score"]} for h in partial_hits],
                    "answer": partial_answer},
    }
    return row


def write_reports(rows: list[dict], excluded_modules: list[str], elapsed_s: float, generate: bool,
                   out_dir: str) -> tuple[str, str]:
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
        f.write(f"Complete: `{DEFAULT_COMPLETE}`  \n")
        f.write(f"Partial: `{DEFAULT_PARTIAL}` (excluded: {', '.join(excluded_modules)})\n\n")
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--complete-chunks", default=DEFAULT_COMPLETE)
    ap.add_argument("--partial-chunks", default=DEFAULT_PARTIAL)
    ap.add_argument("--excluded-modules", nargs="+", default=DEFAULT_EXCLUDED)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="only score the first N gold queries")
    ap.add_argument("--generate", action="store_true",
                     help="also run answer.py's _generate() on each side and check for citations "
                          "the model couldn't have gotten from its own retrieved context")
    ap.add_argument("--rebuild-partial", action="store_true",
                     help="regenerate --partial-chunks (+ its .manifest.json) from --complete-chunks "
                          "before scoring, instead of using whatever's already on disk")
    args = ap.parse_args()

    if args.rebuild_partial or not os.path.exists(args.partial_chunks):
        build_partial_chunks(args.complete_chunks, args.partial_chunks, args.excluded_modules)

    t0 = time.time()
    complete_chunks_data = load_chunks(args.complete_chunks)
    file_by_id = {c["id"]: c["file"] for c in complete_chunks_data}
    complete_bm25 = BM25(complete_chunks_data)
    partial_bm25 = BM25(load_chunks(args.partial_chunks))

    gold_rows = load_jsonl(args.gold)
    if args.limit:
        gold_rows = gold_rows[: args.limit]

    rows = []
    for row in gold_rows:
        gold_ids = row["relevant_chunk_ids"]
        mods = _gold_modules(gold_ids, file_by_id)
        r = score_row(row["question"], gold_ids, mods, row.get("note", ""), args.excluded_modules,
                      complete_bm25, partial_bm25, file_by_id, args.top_k, args.generate)
        rows.append(r)
        print(f"scored: {row['question'][:70]!r}  verdict={r['verdict']}")

    elapsed_s = time.time() - t0
    write_reports(rows, args.excluded_modules, elapsed_s, args.generate, HERE)
    return 0


if __name__ == "__main__":
    sys.exit(main())

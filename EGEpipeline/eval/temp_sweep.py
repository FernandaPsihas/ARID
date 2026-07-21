"""temp_sweep.py -- sweep generation temperature against a handful of gold
questions to see how citation accuracy and answer determinism change.

Different question from bench_retrieval.py (that scores retrieval, not
generation) and bench_ab.py (missing; that scored hallucination on a
degraded index). This scores the generation step itself: for each candidate
temperature, does the model still cite the right file, and how much does its
answer vary rep-to-rep at that temperature? (7/20 meeting item 3: lower
temperature should mean more deterministic answers.)

Retrieval is done once per question (same chunks reused across every
temperature/rep) so temperature is the only thing varying between runs.

Usage:
    python temp_sweep.py                          # default temps/questions/reps
    python temp_sweep.py --temps 0.0 0.3 0.6
    python temp_sweep.py --reps 5 --limit 3
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EGE_ROOT = os.path.dirname(HERE)
DEFAULT_GOLD = os.path.join(HERE, "gold.jsonl")
DEFAULT_CHUNKS = os.path.join(os.path.dirname(EGE_ROOT), "chunks.jsonl")

sys.path.insert(0, EGE_ROOT)
from answer import _retrieve, _generate, TOP_K  # noqa: E402


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_id_to_file(chunks_path: str) -> dict:
    m = {}
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            m[c["id"]] = c["file"]
    return m


def cited(answer_text: str, filepath: str) -> bool:
    """Loose check: does the answer mention this file (by basename)?"""
    return os.path.basename(filepath) in answer_text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--chunks", default=DEFAULT_CHUNKS)
    ap.add_argument("--temps", type=float, nargs="+", default=[0.0, 0.2, 0.5, 0.8])
    ap.add_argument("--reps", type=int, default=3, help="generations per (question, temp)")
    ap.add_argument("--limit", type=int, default=5, help="how many gold questions to sample")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    args = ap.parse_args()

    id_to_file = build_id_to_file(args.chunks)
    gold_rows = load_jsonl(args.gold)[: args.limit]

    print(f"sweeping temps={args.temps} over {len(gold_rows)} questions x {args.reps} reps "
          f"({len(args.temps) * len(gold_rows) * args.reps} generation calls)", file=sys.stderr)

    # retrieve once per question -- same chunks reused at every temp/rep so
    # temperature is the only variable changing between generation calls.
    retrieved = {row["question"]: _retrieve(row["question"], top_k=args.top_k) for row in gold_rows}

    per_temp = {}
    per_query_records = []
    for temp in args.temps:
        accs = []
        agree_flags = []
        sim_scores = []
        for row in gold_rows:
            q = row["question"]
            expected_files = [id_to_file.get(g, "") for g in row["relevant_chunk_ids"]]
            chunks = retrieved[q]
            reps_text, reps_correct = [], []
            for r in range(args.reps):
                t0 = time.time()
                text = _generate(q, chunks, temperature=temp)
                dt = time.time() - t0
                correct = any(cited(text, f) for f in expected_files if f)
                reps_text.append(text)
                reps_correct.append(correct)
                accs.append(correct)
                print(f"  [temp={temp}] {q[:50]!r} rep{r + 1}/{args.reps} "
                      f"correct={correct} ({dt:.1f}s)", file=sys.stderr)
            agree_flags.append(all(reps_correct) or not any(reps_correct))
            pair_sim = None
            if len(reps_text) > 1:
                pair_sims = [
                    difflib.SequenceMatcher(None, reps_text[i], reps_text[i + 1]).ratio()
                    for i in range(len(reps_text) - 1)
                ]
                pair_sim = sum(pair_sims) / len(pair_sims)
                sim_scores.append(pair_sim)
            per_query_records.append({
                "temp": temp, "question": q, "reps_correct": reps_correct,
                "avg_pairwise_similarity": pair_sim,
            })
        per_temp[temp] = {
            "citation_accuracy": sum(accs) / len(accs) if accs else 0.0,
            "determinism_agreement": sum(agree_flags) / len(agree_flags) if agree_flags else 0.0,
            "avg_pairwise_similarity": sum(sim_scores) / len(sim_scores) if sim_scores else None,
        }

    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(HERE, f"temp_sweep_{ts}.json")
    md_path = os.path.join(HERE, f"temp_sweep_{ts}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"gold": args.gold, "temps": args.temps, "reps": args.reps,
                    "n_questions": len(gold_rows), "per_temp": per_temp,
                    "per_query": per_query_records}, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# ARID temperature sweep ({ts})\n\n")
        f.write(f"{len(gold_rows)} questions x {args.reps} reps per temperature\n\n")
        f.write("| temp | citation accuracy | rep agreement | avg pairwise similarity |\n")
        f.write("|---|---|---|---|\n")
        for temp in args.temps:
            s = per_temp[temp]
            sim = f"{s['avg_pairwise_similarity']:.3f}" if s["avg_pairwise_similarity"] is not None else "n/a"
            f.write(f"| {temp} | {s['citation_accuracy']:.2f} | {s['determinism_agreement']:.2f} | {sim} |\n")

    print(f"\nwrote {md_path}\nwrote {json_path}")
    print("per_temp:", json.dumps(per_temp, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

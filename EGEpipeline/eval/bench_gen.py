"""bench_gen.py -- generation speed of the local model: tokens/sec plus a
user-friendly "a typical answer takes ~N s". Answers meeting item #4 (7/31):
CPU vs GPU inference numbers for the generative model.

It measures whatever Ollama host you point --gen-host at, so the SAME script
does both arms -- run it twice and compare:

  GPU: the shared server (default). Read-only; doesn't disturb other users.
  CPU: a throwaway CPU-only Ollama that mounts the SAME model volume, so
       there's no 18GB re-pull and the shared GPU model is left alone:

         docker run -d --name arid-cpu-ollama -p 11435:11434 \\
             -v arid-shared_ollama_data:/root/.ollama ollama/ollama
         # (no --gpus flag => CPU-only)
         python bench_gen.py --gen-host http://localhost:11435 --label cpu
         docker rm -f arid-cpu-ollama

Retrieval (query embedding + Qdrant) always uses the shared services, so the
--gen-host container only ever generates. tok/s comes straight from Ollama's
eval_count/eval_duration (the generation phase only -- load and prompt-eval
are excluded), so it's a clean hardware number regardless of retrieval.

Per the 7/31 decision this is a SHORT run + extrapolation, not a full CPU
benchmark -- default is 3 gold questions. A 30B model on CPU is slow; don't
crank --limit up on the CPU arm.

Usage:
    python bench_gen.py                                 # GPU (shared server), 3 questions
    python bench_gen.py --gen-host http://localhost:11435 --label cpu
    python bench_gen.py --limit 5 --top-k 6
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EGE_ROOT = os.path.dirname(HERE)
DEFAULT_GOLD = os.path.join(HERE, "gold.jsonl")

sys.path.insert(0, EGE_ROOT)
# Reuse the real production prompt/model so the numbers reflect what researchers
# actually pay, not a synthetic prompt. _retrieve gives real retrieved context.
from answer import (GEN_MODEL, NUM_CTX, TEMPERATURE, SYSTEM,  # noqa: E402
                    _format_context, _retrieve)


def _field(resp, key):
    """Ollama's ChatResponse is dict-like in some client versions and a pydantic
    object in others -- read the timing fields either way."""
    try:
        return resp[key]
    except (TypeError, KeyError):
        return getattr(resp, key, None)


def bench(gen_host: str, questions: list[str], top_k: int) -> list[dict]:
    from ollama import Client  # lazy: only the generation arm needs it
    client = Client(host=gen_host)

    # Warm the model once so the first real sample doesn't eat the cold load time
    # (load_duration is separate from eval_duration, but warming keeps latency
    # numbers honest too, and forces a clear failure here if the host is wrong).
    client.chat(model=GEN_MODEL, messages=[{"role": "user", "content": "hi"}],
                options={"num_ctx": NUM_CTX, "temperature": TEMPERATURE})

    rows = []
    for i, q in enumerate(questions, 1):
        chunks = _retrieve(q, top_k)  # shared services embed the query + Qdrant
        prompt = f"Question: {q}\n\nCode snippets:\n\n{_format_context(chunks)}"
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt}]
        resp = client.chat(model=GEN_MODEL, messages=messages,
                           options={"num_ctx": NUM_CTX, "temperature": TEMPERATURE})

        eval_count = _field(resp, "eval_count") or 0
        eval_dur = _field(resp, "eval_duration") or 0        # ns, generation only
        prompt_dur = _field(resp, "prompt_eval_duration") or 0  # ns, prefill
        if not eval_count or not eval_dur:
            print(f"  [{i}/{len(questions)}] no timing returned, skipping: {q[:50]!r}",
                  file=sys.stderr)
            continue
        gen_s = eval_dur / 1e9
        tok_s = eval_count / gen_s
        answer_s = (eval_dur + prompt_dur) / 1e9  # warm perceived time: prefill + generate
        rows.append({"question": q, "gen_tokens": eval_count, "tok_per_s": tok_s,
                     "gen_s": gen_s, "answer_s": answer_s})
        print(f"  [{i}/{len(questions)}] {tok_s:6.1f} tok/s  "
              f"{eval_count:4d} tok in {gen_s:5.1f}s  ({q[:45]!r})")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                     help="Ollama host to GENERATE against (GPU=shared default, CPU=your temp container)")
    ap.add_argument("--label", default="gpu", help="tag for the report file (e.g. gpu, cpu)")
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--limit", type=int, default=3, help="how many gold questions (short run -- see docstring)")
    ap.add_argument("--top-k", type=int, default=6, help="context chunks per question (matches answer.py TOP_K)")
    args = ap.parse_args()

    with open(args.gold, encoding="utf-8") as f:
        questions = [json.loads(line)["question"] for line in f if line.strip()][: args.limit]

    print(f"benchmarking {GEN_MODEL} on {args.gen_host} ({args.label}) "
          f"over {len(questions)} questions", file=sys.stderr)
    rows = bench(args.gen_host, questions, args.top_k)
    if not rows:
        sys.exit("FATAL: no timed generations -- is --gen-host reachable and the model pulled there?")

    med_tok_s = statistics.median(r["tok_per_s"] for r in rows)
    med_tokens = statistics.median(r["gen_tokens"] for r in rows)
    med_answer_s = statistics.median(r["answer_s"] for r in rows)

    print(f"\n=== {args.label} ({args.gen_host}) ===")
    print(f"median generation speed : {med_tok_s:.1f} tok/s")
    print(f"median answer length    : {med_tokens:.0f} tokens")
    print(f"typical answer time     : ~{med_answer_s:.0f} s "
          f"(a ~{med_tokens:.0f}-token answer at {med_tok_s:.0f} tok/s)")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(HERE, f"gen_report_{args.label}_{ts}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": GEN_MODEL, "gen_host": args.gen_host, "label": args.label,
                   "median_tok_per_s": med_tok_s, "median_gen_tokens": med_tokens,
                   "median_answer_s": med_answer_s, "per_query": rows}, f, indent=2)
    print(f"\nwrote {out}")
    return 0


def _selfcheck():
    # tok/s math: 100 tokens generated in 2.0s (2e9 ns) -> 50 tok/s.
    fake = {"eval_count": 100, "eval_duration": 2_000_000_000,
            "prompt_eval_duration": 500_000_000}
    ec = _field(fake, "eval_count")
    tok_s = ec / (_field(fake, "eval_duration") / 1e9)
    answer_s = (_field(fake, "eval_duration") + _field(fake, "prompt_eval_duration")) / 1e9
    assert tok_s == 50.0, tok_s
    assert answer_s == 2.5, answer_s
    # dict-vs-object field access degrades to None, never throws
    assert _field(fake, "missing") is None
    print("ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        _selfcheck()
        sys.exit(0)
    sys.exit(main())

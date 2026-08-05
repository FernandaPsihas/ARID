"""stress_test.py -- how many concurrent researchers can the live stack actually
serve before latency blows up or requests start failing? Ramps the number of
simultaneous answer() calls and watches per-request latency, success rate, and
GPU VRAM/util at each level.

Different question from bench_gen.py (single-stream tok/s, one request at a time)
and temp_sweep.py (generation determinism vs temperature): this one is purely
about CONCURRENCY. docker-compose.shared.yml sets OLLAMA_NUM_PARALLEL=8, so the
interesting questions are (a) does concurrency 10 -- the actual target -- hold,
and (b) what happens past 8, where Ollama has to queue.

Every worker calls the real production path, answer() (hybrid retrieval + local
qwen3-coder:30b), with real questions pulled from gold.jsonl, so the numbers are
what researchers would actually feel. A background thread polls nvidia-smi each
second so VRAM/util can be lined up against the request timeline per level.

A request that hangs can't wedge the harness: each future has a hard wall-clock
timeout (--timeout, default 120s given a ~19s baseline). answer() swallows
GenerationUnavailable into a "[generation unavailable]" string, so that's
detected and counted as a failure too, not a fake success.

Usage:
    python stress_test.py --test                 # pure-math selfcheck, no network
    python stress_test.py                         # full ramp 1/3/5/8/10/12
    python stress_test.py --levels 1 5 10         # custom concurrency levels
    python stress_test.py --limit 1               # smoke: only the level 1 baseline
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

HERE = os.path.dirname(os.path.abspath(__file__))
EGE_ROOT = os.path.dirname(HERE)
DEFAULT_GOLD = os.path.join(HERE, "gold.jsonl")

sys.path.insert(0, EGE_ROOT)
from env_setup import ensure_env  # noqa: E402
ensure_env()

import json  # noqa: E402  (stdlib; imported after env guard to match sibling scripts)
from metrics import load_jsonl  # noqa: E402  shared eval-script JSONL loader
from answer import answer  # noqa: E402  the real production retrieval+generation path

GEN_UNAVAILABLE_MARK = "[generation unavailable]"  # answer() emits this instead of raising


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile (no interpolation, no numpy). p=50 -> median-ish,
    p=95 -> the value at/above which 5% of samples sit. Returns None if empty."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = math.ceil(pct / 100.0 * len(s))  # 1-based rank
    rank = max(1, min(rank, len(s)))
    return s[rank - 1]


def summarize(records: list[dict]) -> dict:
    """p50/p95/max latency, success rate and counts over one level's requests."""
    lats = [r["latency"] for r in records if r["success"]]
    n = len(records)
    ok = sum(1 for r in records if r["success"])
    return {
        "requests": n,
        "successes": ok,
        "failures": n - ok,
        "success_rate": ok / n if n else 0.0,
        "p50_latency": percentile(lats, 50),
        "p95_latency": percentile(lats, 95),
        "max_latency": max(lats) if lats else None,
        "mean_latency": statistics.mean(lats) if lats else None,
    }


class GpuMonitor:
    """Background nvidia-smi poller. Samples (t, mem_used_mb, util_pct) every
    `interval` seconds until stopped. Degrades to no-op if nvidia-smi is absent."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll_once(self) -> None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            line = out.stdout.strip().splitlines()[0]
            mem, util = (x.strip() for x in line.split(","))
            self.samples.append({"t": time.time(), "mem_used_mb": int(mem),
                                 "util_pct": int(util)})
        except Exception as e:  # nvidia-smi missing / unparsable -- record and move on
            self.samples.append({"t": time.time(), "error": f"{type(e).__name__}: {e}"})

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._poll_once()
            self._stop.wait(self.interval)

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 5)

    def peak_mem(self) -> int | None:
        mems = [s["mem_used_mb"] for s in self.samples if "mem_used_mb" in s]
        return max(mems) if mems else None


def _one_request(idx: int, question: str, timeout: float) -> dict:
    """Run one answer() call, timing it and catching everything so a single
    failure never kills the batch. Timeout is enforced by the caller via
    Future.result(); this only handles exceptions answer() might raise."""
    start = time.time()
    try:
        text = answer(question)
        latency = time.time() - start
        if GEN_UNAVAILABLE_MARK in text:
            return {"idx": idx, "start": start, "end": time.time(), "latency": latency,
                    "success": False, "error_type": "GenerationUnavailable",
                    "error": text.split(GEN_UNAVAILABLE_MARK, 1)[1].strip()[:200]}
        return {"idx": idx, "start": start, "end": time.time(), "latency": latency,
                "success": True, "error_type": None, "error": None,
                "answer_chars": len(text)}
    except Exception as e:  # connection reset, qdrant down, etc -- one request only
        return {"idx": idx, "start": start, "end": time.time(),
                "latency": time.time() - start, "success": False,
                "error_type": type(e).__name__, "error": str(e)[:200]}


def run_level(level: int, questions: list[str], timeout: float,
              gpu_interval: float) -> dict:
    """Fire `level` answer() calls concurrently, one per worker, with a GPU
    monitor running for the duration. Returns records + summary + gpu series."""
    gpu = GpuMonitor(gpu_interval)
    gpu.start()
    t0 = time.time()
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=level) as ex:
        # submit_time lets us reconstruct a record for a request that never
        # returns within the wall-clock timeout below.
        futures = {}
        for i in range(level):
            q = questions[i % len(questions)]
            futures[ex.submit(_one_request, i, q, timeout)] = (i, time.time())
        # Per-request wall-clock timeout measured from batch start (all workers
        # start together since workers==tasks). A hung request is recorded as a
        # timeout failure; its thread is abandoned (daemon), never joined.
        # ponytail: abandoned threads leak until process exit -- fine for a
        # one-shot harness; use a subprocess pool if this ever runs as a service.
        deadline = t0 + timeout
        for fut, (i, submitted) in futures.items():
            remaining = max(0.0, deadline - time.time())
            try:
                records.append(fut.result(timeout=remaining))
            except FutureTimeout:
                records.append({"idx": i, "start": submitted, "end": time.time(),
                                "latency": time.time() - submitted, "success": False,
                                "error_type": "Timeout",
                                "error": f"exceeded {timeout}s wall-clock"})
    gpu.stop()
    wall = time.time() - t0
    records.sort(key=lambda r: r["idx"])
    summary = summarize(records)
    summary["wall_s"] = wall
    summary["peak_mem_mb"] = gpu.peak_mem()
    return {"level": level, "records": records, "summary": summary,
            "gpu_samples": gpu.samples}


def _verify_stack() -> None:
    """Hard-fail early if the live stack isn't up -- matches how the other eval
    scripts refuse to produce numbers against a half-up system. Does a single
    real answer() call as the end-to-end smoke test (retrieval + generation)."""
    print("verifying live stack (one real answer() call)...", file=sys.stderr)
    try:
        text = answer("What does DUNEAnaEventUtils::HasNeutrino check?")
    except Exception as e:
        sys.exit(f"FATAL: stack not answering -- {type(e).__name__}: {e}\n"
                 "Are Qdrant (:6333) and Ollama (:11434) both up?")
    if GEN_UNAVAILABLE_MARK in text:
        sys.exit(f"FATAL: generation unavailable -- {text[:200]}\n"
                 "Retrieval works but Ollama/qwen3-coder:30b isn't serving.")
    print("  stack OK, starting ramp.", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 3, 5, 8, 10, 12],
                    help="concurrency levels to ramp through (default matches OLLAMA_NUM_PARALLEL=8 and the 10-user target)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N levels (smoke test, e.g. --limit 1)")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="per-request wall-clock timeout in seconds (~19s baseline)")
    ap.add_argument("--gpu-interval", type=float, default=1.0,
                    help="nvidia-smi poll interval in seconds")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the pre-run live-stack smoke check")
    args = ap.parse_args()

    levels = args.levels[: args.limit] if args.limit else args.levels
    questions = [row["question"] for row in load_jsonl(args.gold)]
    if not questions:
        sys.exit(f"FATAL: no questions in {args.gold}")

    if not args.no_verify:
        _verify_stack()

    print(f"\nramping concurrency {levels} over {len(questions)} gold questions "
          f"(timeout {args.timeout}s/request)\n", file=sys.stderr)
    hdr = f"{'level':>5} {'ok/req':>8} {'succ%':>6} {'p50':>7} {'p95':>7} {'max':>7} {'wall':>7} {'peakVRAM':>9}"
    print(hdr)
    print("-" * len(hdr))

    per_level = []
    for level in levels:
        res = run_level(level, questions, args.timeout, args.gpu_interval)
        per_level.append(res)
        s = res["summary"]
        def f(x):
            return f"{x:.1f}" if isinstance(x, (int, float)) else "n/a"
        vram = f"{s['peak_mem_mb']}MB" if s["peak_mem_mb"] else "n/a"
        print(f"{level:>5} {s['successes']}/{s['requests']:<6} "
              f"{s['success_rate']*100:>5.0f}% {f(s['p50_latency']):>7} "
              f"{f(s['p95_latency']):>7} {f(s['max_latency']):>7} "
              f"{s['wall_s']:>6.1f}s {vram:>9}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(HERE, f"stress_report_{ts}.json")
    with open(out, "w", encoding="utf-8") as fp:
        json.dump({"gold": args.gold, "levels": levels, "timeout_s": args.timeout,
                   "n_questions": len(questions), "per_level": per_level}, fp, indent=2)
    print(f"\nwrote {out}")
    return 0


def _selfcheck():
    # percentile: nearest-rank on a known series 1..10.
    vals = list(range(1, 11))
    assert percentile(vals, 50) == 5, percentile(vals, 50)     # ceil(.5*10)=5 -> vals[4]=5
    assert percentile(vals, 95) == 10, percentile(vals, 95)    # ceil(.95*10)=10 -> vals[9]=10
    assert percentile(vals, 100) == 10
    assert percentile([42], 95) == 42          # single sample
    assert percentile([], 50) is None          # empty -> None

    # summarize: 3 ok (latencies 10/20/30) + 1 failure -> 75% success, correct stats.
    recs = [
        {"latency": 10.0, "success": True},
        {"latency": 20.0, "success": True},
        {"latency": 30.0, "success": True},
        {"latency": 5.0, "success": False},   # failures excluded from latency stats
    ]
    s = summarize(recs)
    assert s["requests"] == 4 and s["successes"] == 3 and s["failures"] == 1, s
    assert s["success_rate"] == 0.75, s
    assert s["p50_latency"] == 20.0, s        # ceil(.5*3)=2 -> [10,20,30][1]=20
    assert s["max_latency"] == 30.0, s
    assert s["mean_latency"] == 20.0, s

    # all-failure level: no latency stats, zero success rate, never divides by zero.
    s2 = summarize([{"latency": 1.0, "success": False}])
    assert s2["success_rate"] == 0.0 and s2["p50_latency"] is None, s2

    print("ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        _selfcheck()
        sys.exit(0)
    sys.exit(main())

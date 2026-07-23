"""plot_embed_bakeoff.py -- plot the qwen3-embedding:0.6b / 8b / nomic-embed-code
retrieval bake-off (7/20 meeting item 7, due 7/24). Reads three bench_retrieval.py
JSON reports (one per embed model, run against the same gold.jsonl/collection setup
on QuarkyLab 7/21) and plots recall/precision/nDCG@k, k on the x-axis, one line per
model, so the three models can actually be compared instead of just eyeballed as
separate aggregate dicts.

Usage:
    python plot_embed_bakeoff.py
    python plot_embed_bakeoff.py --out bakeoff.png
"""

import argparse
import json
import os

import matplotlib.pyplot as plt

REPORTS = [
    ("retrieval_report_20260721_201724.json", "qwen3-embedding:0.6b"),
    ("retrieval_report_20260721_201747.json", "qwen3-embedding:8b"),
    ("retrieval_report_20260721_201810.json", "nomic-embed-code"),
]
METRICS = ("recall", "precision", "ndcg")


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot(out_path: str) -> None:
    here = os.path.dirname(__file__)
    reports = [(_load(os.path.join(here, fname)), label) for fname, label in REPORTS]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 4.5))
    for ax, metric in zip(axes, METRICS):
        for report, label in reports:
            agg = report["aggregate"][metric]
            ks = sorted(int(k) for k in agg)
            ax.plot(ks, [agg[str(k)] for k in ks], marker="o", label=label)
        ax.set_title(metric)
        ax.set_xlabel("k")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("score")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Embedding bake-off: retrieval quality vs k (gold.jsonl, dunereco)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")

    for report, label in reports:
        print(f"{label:24s} mrr={report['aggregate']['mrr']:.3f}  "
              f"recall@10={report['aggregate']['recall']['10']:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="embed_bakeoff.png")
    args = p.parse_args()
    plot(args.out)

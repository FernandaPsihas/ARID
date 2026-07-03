"""Grade an embedding model on the gold query set: Recall@k + MRR.

A gold row is {"question": str, "relevant_chunk_ids": [str, ...]}. Each model is
scored over ONE Qdrant collection that was indexed with that same model, so the
comparison is fair (you must index the corpus once per model first).

Flow to compare the three candidates:
  1) For each model, index the corpus into its own collection. embed_store.py
     hardcodes EMBED_MODEL / COLLECTION, so either edit those two constants per
     run or set the env vars this script's sibling reads. Suggested names:
        qwen3-embedding:0.6b   -> collection "dunereco_qwen06"
        qwen3-embedding:4b     -> collection "dunereco_qwen4"
        nomic-embed-code       -> collection "dunereco_nomic"
  2) Run this once per model:
        python grade.py --model qwen3-embedding:4b --collection dunereco_qwen4
  3) Compare the printed tables. Higher Recall@k and MRR win.

Needs Ollama (:11434) and Qdrant (:6333) up -- same services the pipeline uses.
"""

import argparse
import json
import os
import sys

import ollama
from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DEFAULT_GOLD = os.path.join(os.path.dirname(__file__), "gold.jsonl")
MAX_EMBED_CHARS = 6000  # match embed_store.py so query encoding is identical


def embed(model: str, text: str) -> list[float]:
    return ollama.embeddings(
        model=model, prompt=text[:MAX_EMBED_CHARS], options={"num_ctx": 8192}
    )["embedding"]


def load_gold(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for r in rows:
        if not r.get("question") or not r.get("relevant_chunk_ids"):
            sys.exit(f"bad gold row (needs question + relevant_chunk_ids): {r}")
    return rows


def grade(model: str, collection: str, gold: list[dict], ks=(1, 5, 10)) -> dict:
    client = QdrantClient(url=QDRANT_URL)
    if not client.collection_exists(collection):
        sys.exit(f"collection '{collection}' not found -- index the corpus with "
                 f"{model} into it first (see this file's header).")
    maxk = max(ks)
    hits_at = {k: 0 for k in ks}
    rr_sum = 0.0
    misses = []
    for row in gold:
        relevant = set(row["relevant_chunk_ids"])
        res = client.query_points(
            collection_name=collection, query=embed(model, row["question"]),
            limit=maxk,
        ).points
        ranked = [p.payload["chunk_id"] for p in res]
        rank = next((i + 1 for i, cid in enumerate(ranked) if cid in relevant), None)
        if rank is None:
            misses.append(row["question"][:70])
        else:
            rr_sum += 1.0 / rank
            for k in ks:
                if rank <= k:
                    hits_at[k] += 1
    n = len(gold)
    return {
        "model": model, "collection": collection, "n": n,
        "recall": {k: hits_at[k] / n for k in ks},
        "mrr": rr_sum / n, "misses": misses,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ollama embedding model name")
    ap.add_argument("--collection", required=True, help="qdrant collection for that model")
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    a = ap.parse_args()

    r = grade(a.model, a.collection, load_gold(a.gold), ks=tuple(a.k))
    print(f"\n{r['model']}  (collection={r['collection']}, {r['n']} queries)")
    for k in a.k:
        print(f"  Recall@{k:<2} {r['recall'][k]:.3f}")
    print(f"  MRR      {r['mrr']:.3f}")
    if r["misses"]:
        print(f"  missed ({len(r['misses'])}): " + "; ".join(r["misses"]))


if __name__ == "__main__":
    main()

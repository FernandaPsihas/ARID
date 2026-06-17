"""BM25 keyword search over chunks.jsonl (stdlib only)."""

import json
import math
import re
import sys
from collections import Counter

K1, B = 1.5, 0.75

# splits code identifiers too: GetHits -> get, hits ; DUNEAnaUtils -> dune, ana, utils
_TOKEN = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class BM25:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.docs = [tokenize(c["text"] + " " + c["symbol"]) for c in chunks]
        self.tf = [Counter(d) for d in self.docs]
        self.len = [len(d) for d in self.docs]
        self.avglen = sum(self.len) / len(self.len) if self.docs else 0
        df = Counter(t for tf in self.tf for t in tf)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if not f:
                    continue
                denom = f + K1 * (1 - B + B * self.len[i] / self.avglen)
                s += self.idf[t] * f * (K1 + 1) / denom
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        out = []
        for s, i in scores[:top_k]:
            c = self.chunks[i]
            out.append({
                "score": s, "chunk_id": c["id"], "file": c["file"],
                "start_line": c["start_line"], "end_line": c["end_line"],
                "symbol": c["symbol"], "language": c["language"], "text": c["text"],
            })
        return out


def load_chunks(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _selfcheck():
    docs = [
        {"id": "a", "file": "a.cc", "start_line": 1, "end_line": 2, "symbol": "GetHits", "language": "cpp", "text": "return GetHits cluster"},
        {"id": "b", "file": "b.cc", "start_line": 1, "end_line": 2, "symbol": "HasNeutrino", "language": "cpp", "text": "bool neutrino particle loop neutrino"},
    ]
    bm = BM25(docs)
    assert tokenize("GetHits") == ["get", "hits"]
    hits = bm.search("neutrino")
    assert hits and hits[0]["chunk_id"] == "b", hits
    assert not bm.search("nonexistentword")
    print("ok")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "--test":
        _selfcheck()
        sys.exit(0)
    chunks = load_chunks(sys.argv[1])
    bm = BM25(chunks)
    query = " ".join(sys.argv[2:]) or "electron lifetime calibration"
    print(f"Loaded {len(chunks)} chunks. Query: {query!r}\n")
    for r in bm.search(query, top_k=10):
        print(f"  [{r['score']:.3f}]  {r['file']}  L{r['start_line']}-{r['end_line']}  {r['symbol']}")

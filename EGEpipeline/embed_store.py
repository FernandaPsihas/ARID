import concurrent.futures
import json
import sys
import os
import uuid

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

sys.path.append(os.path.join(os.path.dirname(__file__), "..")) # allow import from parent dir (testing for demo, we can yeet this later)
sys.path.append(os.path.dirname(__file__))  # sibling qdrant_index

# we need that so we can do this
from chunk_schema import validate_chunk
import qdrant_index as qi

EMBED_MODEL = "qwen3-embedding:0.6b"
# Reads target this name. It's the ALIAS that qdrant_index maintains -- Qdrant
# resolves it to whichever physical collection is currently live, so re-indexing
# can swap the data underneath queries without any reader noticing. (Kept named
# COLLECTION for backwards-compat with anything importing it.)
COLLECTION  = qi.ALIAS
QDRANT_URL  = os.environ.get("QDRANT_URL", "http://localhost:6333")  # service name in docker, localhost otherwise
BATCH_SIZE  = 32

# don't ask me what this is because i dont know
_client = QdrantClient(url=QDRANT_URL)

# ponytail: embeddings are ONE vector for the whole input — passage-sized chunks
# give sharp vectors, whole-file blobs give mushy ones. 99% of chunks are <600
# chars. Truncate the embed INPUT (full text still goes to the payload + BM25, so
# nothing's lost for retrieval). Char->token ratio varies: code ~3:1, but jsonnet
# numeric data tables ~1:1, so 6000 chars worst-case ~6000 tokens — num_ctx=8192
# holds that with margin.
MAX_EMBED_CHARS = 6000

# chunks are embedded BATCH_SIZE at a time in one request; WORKERS such batched
# requests run concurrently so the GPU has back-to-back work queued instead of
# idling between round trips. Match OLLAMA_NUM_PARALLEL on the server
# (docker-compose.shared.yml) or these just queue up there instead of running concurrently.
WORKERS = int(os.environ.get("EMBED_WORKERS", 8))

#couple helpers
def _embed_batch(texts: list[str]) -> list[list[float]]:
    # /api/embed (ollama.embed) takes a list and runs it as one batched forward
    # pass -- much less per-item overhead than N calls to the legacy single-prompt
    # /api/embeddings (ollama.embeddings).
    resp = ollama.embed(model=EMBED_MODEL, input=[t[:MAX_EMBED_CHARS] for t in texts],
                         options={"num_ctx": 8192})
    return resp.embeddings

def _embed(text: str) -> list[float]:
    return _embed_batch([text])[0]

def _dim() -> int:
    return len(_embed("probe"))

def _chunk_id_to_uuid(chunk_id: str) -> str:
    # qdrant only takes uuids, so we do a little conversion
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

# indexing
def index_into(collection: str, chunks: list[dict]):
    # embed + store (batched) into an already-created `collection`. Callers only
    # ever pass a fresh throwaway collection here (see rebuild), never the live
    # one, so a failure or interruption can't damage what queries are reading.
    total = len(chunks)
    failed = []
    done = 0

    def _embed_group(idxs: list[int]) -> list:
        # one batched request for the whole group; if the batch call itself
        # fails, fall back to per-chunk requests so one bad chunk (e.g. a
        # pathological encoding) doesn't sink its whole batch.
        try:
            return _embed_batch([chunks[i]["text"] for i in idxs])
        except Exception:
            results = []
            for i in idxs:
                try:
                    results.append(_embed(chunks[i]["text"]))
                except Exception as e:
                    results.append(e)
            return results

    groups = [list(range(i, min(i + BATCH_SIZE, total))) for i in range(0, total, BATCH_SIZE)]

    # each group is one batched embed call; WORKERS groups run concurrently so
    # the GPU stays fed between requests instead of idling on network/JSON overhead.
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_embed_group, g): g for g in groups}
        for future in concurrent.futures.as_completed(futures):
            idxs = futures[future]
            results = future.result()
            points = []
            for i, r in zip(idxs, results):
                chunk = chunks[i]
                done += 1
                if isinstance(r, Exception):
                    print(f"  ERROR chunk[{i}] {chunk['id']} ({len(chunk['text'])} chars): {r}")
                    failed.append({"index": i, "id": chunk["id"], "file": chunk["file"],
                                   "chars": len(chunk["text"]), "error": str(r)})
                    continue
                points.append(
                    PointStruct(
                        id=_chunk_id_to_uuid(chunk["id"]),
                        vector=r,
                        payload={
                            "chunk_id":   chunk["id"],
                            "file":       chunk["file"],
                            "start_line": chunk["start_line"],
                            "end_line":   chunk["end_line"],
                            "symbol":     chunk["symbol"],
                            "language":   chunk["language"],
                            "text":       chunk["text"],
                        },
                    )
                )

            if points:
                _client.upsert(collection_name=collection, points=points)
            print(f"  {done}/{total} chunks indexed")

    if failed:
        import json as _json
        with open("embed_errors.json", "w") as f:
            _json.dump(failed, f, indent=2)
        print(f"\n{len(failed)} chunks failed — see embed_errors.json")
    print(f"\nDone. {total - len(failed)}/{total} chunks stored in {collection}.")


def rebuild(chunks: list[dict], *, force: bool = False) -> str | None:
    """Validate + (re)index the corpus into the shared store, safely.

    Delegates the concurrency-safe dance (build a fresh uniquely-named collection,
    atomically swap the alias, clean up orphans) to qdrant_index.rebuild; this just
    supplies the embed step and up-front validation. Returns the live collection
    name, or None if the build was skipped because the index already matches.
    """
    bad = [(i, errs) for i, c in enumerate(chunks) if (errs := validate_chunk(c))]
    if bad:
        print(f"Validation failed on {len(bad)} chunk(s):")
        for i, errs in bad:
            print(f"  chunk[{i}] (id={chunks[i].get('id', '?')}): {errs}")
        sys.exit(1)

    dim = _dim()
    return qi.rebuild(
        _client, chunks, dim,
        embed_into=lambda collection: index_into(collection, chunks),
        force=force,
    )

def search_dense(query: str, top_k: int = 20) -> list[dict]:
    hits = _client.query_points(
        collection_name=COLLECTION,
        query=_embed(query),
        limit=top_k,
    ).points
    return [
        {
            "score":      h.score,
            "chunk_id":   h.payload["chunk_id"],
            "file":       h.payload["file"],
            "start_line": h.payload["start_line"],
            "end_line":   h.payload["end_line"],
            "symbol":     h.payload["symbol"],
            "language":   h.payload["language"],
            "text":       h.payload["text"],
        }
        for h in hits
    ]

#cli
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if not args:
        print("Usage: python embed_store.py <chunks.jsonl> [--force]")
        print("  --force  rebuild even if the store already matches these chunks")
        sys.exit(1)

    with open(args[0], encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]  # JSONL, matches extract.py output

    print(f"Loaded {len(chunks)} chunks from {args[0]}")
    # FORCE_EMBED stays supported for the bootstrap/env path; --force is the CLI equivalent.
    force = "--force" in flags or os.environ.get("FORCE_EMBED") == "1"
    live = rebuild(chunks, force=force)
    if live is None:
        sys.exit(0)  # skipped: index already matches the corpus

    test_query = "electron lifetime calibration"
    print(f"\nTest query:'{test_query}':")
    for r in search_dense(test_query, top_k=3):
        print(f"  [{r['score']:.4f}]  {r['file']}  L{r['start_line']}-{r['end_line']}  {r['symbol']}")


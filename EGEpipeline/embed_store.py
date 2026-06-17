import json
import sys
import os
import uuid

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

sys.path.append(os.path.join(os.path.dirname(__file__), "..")) # allow import from parent dir (testing for demo, we can yeet this later)

# we need that so we can do this
from chunk_schema import validate_chunk

EMBED_MODEL = "qwen3-embedding:0.6b"
COLLECTION  = "dunereco"
QDRANT_URL  = "http://localhost:6333"
BATCH_SIZE  = 32

# don't ask me what this is because i dont know
_client = QdrantClient(url=QDRANT_URL)

#couple helpers
def _embed(text: str) -> list[float]:
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]

def _dim() -> int:
    return len(_embed("probe"))

def _chunk_id_to_uuid(chunk_id: str) -> str:
    # qdrant only takes uuids, so we do a little conversion
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

#into setup

def setup():
    # ready collection, delete if its alr there, demo
    vector_dim = _dim()
    if _client.collection_exists(COLLECTION):
        _client.delete_collection(COLLECTION)
    _client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION}' ready  (dim={vector_dim})")


# indexing
def index_chunks(chunks: list[dict]):
    # validate, embed, store (batched)
    bad = [(i, errs) for i, c in enumerate(chunks) if (errs := validate_chunk(c))]
    if bad:
        print(f"Validation failed on {len(bad)} chunk(s):")
        for i, errs in bad:
            print(f"  chunk[{i}] (id={chunks[i].get('id', '?')}): {errs}")
        sys.exit(1)
 
    total = len(chunks)
    batch = []
 
    for i, chunk in enumerate(chunks):
        vector = _embed(chunk["text"])
        batch.append(
            PointStruct(
                id=_chunk_id_to_uuid(chunk["id"]),
                vector=vector,
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
 
        if len(batch) >= BATCH_SIZE or i == total - 1:
            _client.upsert(collection_name=COLLECTION, points=batch)
            print(f"  {i + 1}/{total} chunks indexed")
            batch = []
 
    print(f"\nDone. {total} chunks stored in Qdrant.")

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
    if len(sys.argv) < 2:
        print("Usage: python embed_store.py <chunks.jsonl>")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]  # JSONL, matches extract.py output
 
    print(f"Loaded {len(chunks)} chunks from {sys.argv[1]}")
    setup()
    index_chunks(chunks)
 
    test_query = "electron lifetime calibration"
    print(f"\nTest query:'{test_query}':")
    for r in search_dense(test_query, top_k=3):
        print(f"  [{r['score']:.4f}]  {r['file']}  L{r['start_line']}-{r['end_line']}  {r['symbol']}")


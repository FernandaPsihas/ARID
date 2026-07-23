"""Rebuild a local chunks.jsonl from the shared Qdrant, for the BM25 half.

Retrieval is hybrid: the dense half reads Qdrant over HTTP (nothing local
needed), but BM25 reads a local chunks.jsonl. Producing that file with
extract.py means cloning dunereco and installing the tree-sitter grammars --
which need a C compiler, and it's the one heavy step in an otherwise
pip-install-and-go setup.

The full chunk text already lives in the Qdrant payload, though, so we can just
pull it back out: same content, no clone, no compiler, no shared filesystem.
Needs only qdrant-client, which the query path already depends on.

Usage:
    python EGEpipeline/sync_chunks.py            # writes ./chunks.jsonl
    python EGEpipeline/sync_chunks.py out.jsonl
"""

import json
import os
import sys

CHUNKS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chunks.jsonl")
BATCH = 1000

# payload key -> chunk_schema field. The payload stores the id under "chunk_id"
# and the line numbers as strings, so this isn't a straight copy.
_INT_FIELDS = ("start_line", "end_line")


def _to_chunk(payload: dict) -> dict:
    chunk = {
        "id":       payload["chunk_id"],
        "file":     payload["file"],
        "symbol":   payload["symbol"],
        "language": payload["language"],
        "text":     payload["text"],
    }
    for f in _INT_FIELDS:
        chunk[f] = int(payload[f])
    return chunk


def fetch_chunks(client, collection: str) -> list[dict]:
    """Scroll the whole collection, payloads only -- vectors would be 1024 floats
    per chunk of pure waste, since we only ever feed this to BM25."""
    chunks, offset = [], None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=BATCH, offset=offset,
            with_payload=True, with_vectors=False,
        )
        chunks.extend(_to_chunk(p.payload) for p in points)
        if offset is None:
            break
    return chunks


def write_chunks(chunks: list[dict], path: str) -> None:
    """Atomic: a half-written chunks.jsonl would silently degrade every future
    BM25 search rather than fail loudly."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    os.replace(tmp, path)


def _build(path: str) -> int:
    """Fetch the collection from Qdrant and write chunks.jsonl. Raises on a
    connection/read error, ValueError if the collection is empty. The two
    entry points below wrap this: sync() exits on failure (CLI), ensure_chunks()
    swallows it (query path, must not crash a query that can still run dense)."""
    from qdrant_client import QdrantClient  # lazy: keeps _selfcheck dependency-free

    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    collection = os.environ.get("QDRANT_COLLECTION", "dunereco")
    client = QdrantClient(url=url)
    chunks = fetch_chunks(client, collection)
    if not chunks:
        raise ValueError(f"{collection!r} is empty -- has it been provisioned?")
    write_chunks(chunks, path)
    return len(chunks)


def sync(path: str = CHUNKS_PATH) -> int:
    """CLI entry: build chunks.jsonl or exit(1) with a diagnostic."""
    try:
        n = _build(path)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        print(f"error: couldn't read from {url} ({type(e).__name__}: {e})\n"
              f"Is the shared Qdrant up, and is QDRANT_URL correct?", file=sys.stderr)
        sys.exit(1)
    print(f"wrote {n} chunks to {path}")
    return n


def ensure_chunks(path: str = CHUNKS_PATH) -> bool:
    """Build chunks.jsonl from Qdrant if it's missing; return True if the file is
    usable afterward. Never raises or exits -- the query path calls this so a
    fresh clone gets the BM25 half automatically on its first query, but a failure
    (Qdrant unreachable, empty collection) just degrades to dense-only."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    try:
        n = _build(path)
    except Exception as e:
        print(f"warning: couldn't auto-build the BM25 index from Qdrant "
              f"({type(e).__name__}: {e}); running dense-only", file=sys.stderr)
        return False
    print(f"auto-synced {n} chunks -> {path} (BM25 enabled)", file=sys.stderr)
    return True


def _selfcheck():
    p = {"chunk_id": "a.cc_Foo_1", "file": "a.cc", "start_line": "1", "end_line": "9",
         "symbol": "Foo", "language": "cpp", "text": "body"}
    c = _to_chunk(p)
    assert c["id"] == "a.cc_Foo_1", c
    assert c["start_line"] == 1 and c["end_line"] == 9, c   # strings coerced to int

    sys.path.append(os.path.dirname(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from chunk_schema import validate_chunk
    assert validate_chunk(c) == [], validate_chunk(c)       # round-trips to a valid chunk
    print("ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        _selfcheck()
        sys.exit(0)
    sync(sys.argv[1] if len(sys.argv) > 1 else CHUNKS_PATH)

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

# MUST match whatever model actually embedded the live ALIAS collection -- Qdrant
# rejects a query vector whose dimension doesn't match the collection's (a real
# outage hit 7/23 flipping the alias to nomic-embed-code while this default was
# still qwen3-embedding:0.6b: "expected dim: 3584, got 1024" on every unoverridden
# query). There's no way to derive this automatically from the alias target, so
# it's on whoever runs embed_store.py's rebuild()/CLI with a different
# --embed-model to update this default in the same change.
EMBED_MODEL = os.environ.get("ARID_EMBED_MODEL", "manutic/nomic-embed-code:latest")
# Reads target the shared ALIAS by default -- qdrant_index maintains it, and Qdrant
# resolves it to whichever physical collection is currently live, so re-indexing can
# swap the data underneath queries without any reader noticing (see qdrant_index.py).
# ARID_EMBED_COLLECTION still overrides this, for eval work that needs to query a
# specific bake-off collection directly instead of the live alias (e.g.
# eval/bench_retrieval.py's --collection flag).
COLLECTION  = os.environ.get("ARID_EMBED_COLLECTION", qi.ALIAS)
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

def _point(chunk: dict, vector) -> PointStruct:
    # Built by both paths -- freshly embedded chunks and chunks whose vector was
    # reused from the live index (see _reuse_cached) -- so the payload can't drift
    # between them.
    return PointStruct(
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
            # Which model produced `vector`. Only read by --incremental, but written
            # always: a vector is meaningless to any other model, and without this
            # recorded per-point there is no way to tell whether reusing one is safe.
            "embed_model": EMBED_MODEL,
        },
    )

def _reuse_cached(collection: str, chunks: list[dict], by_text: dict[str, list[int]]) -> int:
    """Copy vectors for already-embedded text out of the live index into `collection`.

    Embedding is deterministic, so any chunk whose embed input already exists in the
    live index doesn't need the GPU again -- we can lift its vector straight across.
    Adding a repo or editing a handful of files changes a small fraction of the
    corpus, so this turns a ~80 minute rebuild into a few minutes of the genuinely
    new chunks.

    DESTRUCTIVE on `by_text`: every text covered here is popped, so what survives is
    exactly the set that still needs embedding. Returns the number of chunks covered.

    Streams a page at a time rather than building a {text: vector} map first --
    43k points at 3584 dims is ~600MB of float, which is not worth holding just to
    copy it straight back out.
    """
    live = qi.resolve_alias(_client)
    if live is None:
        print("  incremental: nothing provisioned yet, embedding everything")
        return 0

    reused = 0
    skipped_model = 0
    offset = None
    while by_text:
        points, offset = _client.scroll(
            collection_name=live, limit=512, offset=offset,
            with_payload=["text", "embed_model"], with_vectors=True,
        )
        batch = []
        for p in points:
            # A vector is only valid for the model that produced it, and a same-dim
            # different-model reuse would sail past Qdrant's dimension check and
            # serve silent garbage. Points written before this field existed have no
            # embed_model, so the first incremental run after this change reuses
            # nothing and re-embeds in full -- it fails closed, then self-heals.
            if p.payload.get("embed_model") != EMBED_MODEL:
                skipped_model += 1
                continue
            # Key on the same truncation _embed_batch applies. The payload holds the
            # FULL text, so truncate here to match what was actually embedded.
            for i in by_text.pop(p.payload["text"][:MAX_EMBED_CHARS], ()):
                batch.append(_point(chunks[i], p.vector))
        if batch:
            _client.upsert(collection_name=collection, points=batch)
            reused += len(batch)
            print(f"  reused {reused} chunks from {live}", flush=True)
        if offset is None:
            break

    if skipped_model:
        print(f"  incremental: {skipped_model} live points were embedded by a "
              f"different model (or before embed_model was recorded) and can't be reused")
    return reused


# indexing
def index_into(collection: str, chunks: list[dict], *, incremental: bool = False):
    # embed + store (batched) into an already-created `collection`. Callers only
    # ever pass a fresh throwaway collection here (see rebuild), never the live
    # one, so a failure or interruption can't damage what queries are reading.
    total = len(chunks)
    failed = []

    # The vector is a pure function of the (truncated) embed input, so identical
    # text embeds once and fans out to every chunk that shares it. 21% of the Tier 1
    # corpus is duplicate text -- C++ forward declarations ("class ParameterSet"
    # appears 85 times across 12 repos) and jsonnet import bindings
    # ("wc = import 'wirecell.jsonnet'", 160 times) that the chunkers emit as
    # standalone chunks. Keyed on the SAME truncation _embed_batch applies, so two
    # texts differing only past MAX_EMBED_CHARS correctly share one vector.
    by_text: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        by_text.setdefault(c["text"][:MAX_EMBED_CHARS], []).append(i)
    if len(by_text) < total:
        print(f"  {total} chunks -> {len(by_text)} distinct embed inputs "
              f"({total - len(by_text)} duplicates skipped)")

    # Opt-in: lift vectors for unchanged text off the live index instead of
    # re-embedding them. Pops what it covers, so `by_text` is left holding only the
    # chunks that actually need the GPU.
    done = _reuse_cached(collection, chunks, by_text) if incremental else 0
    if incremental:
        print(f"  {done} chunks reused, {len(by_text)} distinct texts left to embed")
    texts = list(by_text)

    def _embed_group(group: list[str]) -> list:
        # one batched request for the whole group; if the batch call itself
        # fails, fall back to per-chunk requests so one bad chunk (e.g. a
        # pathological encoding) doesn't sink its whole batch.
        try:
            return _embed_batch(group)
        except Exception:
            results = []
            for t in group:
                try:
                    results.append(_embed(t))
                except Exception as e:
                    results.append(e)
            return results

    groups = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

    # each group is one batched embed call; WORKERS groups run concurrently so
    # the GPU stays fed between requests instead of idling on network/JSON overhead.
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_embed_group, g): g for g in groups}
        for future in concurrent.futures.as_completed(futures):
            group = futures[future]
            results = future.result()
            points = []
            # one embed result fans out to every chunk that shares that text
            for i, r in ((i, r) for text, r in zip(group, results) for i in by_text[text]):
                chunk = chunks[i]
                done += 1
                if isinstance(r, Exception):
                    print(f"  ERROR chunk[{i}] {chunk['id']} ({len(chunk['text'])} chars): {r}")
                    failed.append({"index": i, "id": chunk["id"], "file": chunk["file"],
                                   "chars": len(chunk["text"]), "error": str(r)})
                    continue
                points.append(_point(chunk, r))

            if points:
                _client.upsert(collection_name=collection, points=points)
            print(f"  {done}/{total} chunks indexed")

    if failed:
        import json as _json
        with open("embed_errors.json", "w") as f:
            _json.dump(failed, f, indent=2)
        print(f"\n{len(failed)} chunks failed — see embed_errors.json")
    print(f"\nDone. {total - len(failed)}/{total} chunks stored in {collection}.")


def rebuild(chunks: list[dict], *, force: bool = False,
            incremental: bool = False) -> str | None:
    """Validate + (re)index the corpus into the shared store, safely.

    Delegates the concurrency-safe dance (build a fresh uniquely-named collection,
    atomically swap the alias, clean up orphans) to qdrant_index.rebuild; this just
    supplies the embed step and up-front validation. Returns the live collection
    name, or None if the build was skipped because the index already matches.

    `incremental` reuses vectors for unchanged text from the live index rather than
    re-embedding them. It defaults OFF: while the repo set is still being extended
    and the chunkers are still being changed, a full rebuild is the option that
    can't be subtly wrong, and reindexing is still an occasional admin action
    rather than something on a schedule. The safety model is identical either way
    -- both build a fresh collection and publish by atomic alias swap.
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
        embed_into=lambda collection: index_into(collection, chunks,
                                                 incremental=incremental),
        force=force,
    )

def search_dense(query: str, top_k: int = 20, collection: str | None = None) -> list[dict]:
    """collection overrides the module default (the live shared alias) --
    e.g. for bench_ab.py's partial-index side, which needs its own separate
    collection rather than the production alias."""
    hits = _client.query_points(
        collection_name=collection or COLLECTION,
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
        print("Usage: python embed_store.py <chunks.jsonl> [--force] [--incremental]")
        print("  --force        rebuild even if the store already matches these chunks")
        print("  --incremental  reuse vectors for unchanged text from the live index")
        print("                 instead of re-embedding the whole corpus")
        sys.exit(1)

    with open(args[0], encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]  # JSONL, matches extract.py output

    print(f"Loaded {len(chunks)} chunks from {args[0]}")
    # FORCE_EMBED stays supported for the bootstrap/env path; --force is the CLI equivalent.
    force = "--force" in flags or os.environ.get("FORCE_EMBED") == "1"
    incremental = "--incremental" in flags or os.environ.get("ARID_INCREMENTAL") == "1"
    live = rebuild(chunks, force=force, incremental=incremental)
    if live is None:
        sys.exit(0)  # skipped: index already matches the corpus

    test_query = "electron lifetime calibration"
    print(f"\nTest query:'{test_query}':")
    for r in search_dense(test_query, top_k=3):
        print(f"  [{r['score']:.4f}]  {r['file']}  L{r['start_line']}-{r['end_line']}  {r['symbol']}")


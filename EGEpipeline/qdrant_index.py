"""Concurrency-safe indexing for the ONE shared ARID Qdrant.

The dunereco corpus lives in a single Qdrant that every researcher queries. That
makes re-indexing dangerous if done naively: the old `embed_store.setup()` deleted
the collection and rebuilt it in place, so for the minutes that took, everyone
querying got an empty or half-filled store -- and an interrupted rebuild left it
broken for good. This module removes that hazard by never mutating live data.

How safety is guaranteed (this is the important part):

  * Reads always go through a stable ALIAS ("dunereco"). Qdrant resolves the
    alias to whatever physical collection is currently live.
  * A rebuild embeds into a BRAND-NEW physical collection
    (``dunereco__<fingerprint>__<timestamp>_<random>``) and never touches the
    live one. The random suffix makes the name UNIQUE per rebuild, so two people
    reindexing at the same instant can never land on the same collection.
  * Only once the new collection is fully built is the alias flipped to it, in a
    single atomic ``update_collection_aliases`` call. Readers switch instantly;
    no reader ever observes a partial index.
  * If a rebuild dies partway, the alias never moved -- the live index is still
    the old good one, and the half-built collection is just an orphan that the
    next rebuild sweeps up.

Concurrent rebuilds need no lock. Because each build gets its own unique
collection and only publishes by an atomic alias swap, two rebuilds racing just
each build a complete index and swap: last swap wins, both indexes are complete,
nothing is corrupted. The only cost is the duplicate embed work of the losing
build -- rare (reindex is an admin action) and self-correcting (its collection
is cleaned up as an orphan). Cleanup is age-guarded so it can never delete a
collection another rebuild is still filling.

The public entry point is :func:`rebuild`. Everything else is a helper.
"""

import hashlib
import os
import re
import time
import uuid

from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    Distance,
    PointStruct,
    VectorParams,
)

# Read-side name. Every query targets this; it is either a real collection
# (legacy, pre-migration) or an alias (after the first safe rebuild). Overridable
# so tests can operate on a throwaway name without touching real data.
ALIAS = os.environ.get("QDRANT_COLLECTION", "dunereco")

# Physical collections are "<ALIAS>__<fingerprint>__<epoch>_<random>". The double
# underscore keeps them from ever colliding with a plain alias name; the random
# suffix keeps two same-second rebuilds from colliding with each other. The
# suffix is optional in the regex so pre-random legacy names still parse.
_SEP = "__"
_NAME_RE = re.compile(r"^(?P<alias>.+)__(?P<fp>[0-9a-f]+)__(?P<ts>\d+)(?:_[0-9a-f]+)?$")

# Orphaned physical collections younger than this are left alone, so cleanup can
# never delete a collection another researcher is actively building right now.
ORPHAN_KEEP_SECONDS = int(os.environ.get("ARID_ORPHAN_KEEP", 60 * 60))


def log(msg: str):
    print(f"[index] {msg}", flush=True)


# --- identity & fingerprinting ------------------------------------------------

def fingerprint(chunks: list[dict]) -> str:
    """Stable short hash of the corpus content.

    Same chunks in any order -> same fingerprint, so re-running provision on an
    unchanged corpus is a no-op, while any edit to the chunks forces a rebuild.
    """
    h = hashlib.sha1()
    for cid, text in sorted((c["id"], c["text"]) for c in chunks):
        h.update(cid.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()[:12]


def _physical_name(fp: str) -> str:
    return f"{ALIAS}{_SEP}{fp}{_SEP}{int(time.time())}_{uuid.uuid4().hex[:8]}"


def _parse(name: str) -> tuple[str, int] | None:
    """(fingerprint, ts) for a physical collection we manage, else None."""
    m = _NAME_RE.match(name)
    if not m or m.group("alias") != ALIAS:
        return None
    return m.group("fp"), int(m.group("ts"))


# --- alias resolution ---------------------------------------------------------

def resolve_alias(client) -> str | None:
    """Physical collection the read name currently points at, or None.

    Handles three states: a proper alias (return its target), a legacy plain
    collection still named ALIAS (return ALIAS itself), or nothing provisioned
    yet (return None).
    """
    for a in client.get_aliases().aliases:
        if a.alias_name == ALIAS:
            return a.collection_name
    if client.collection_exists(ALIAS):
        return ALIAS  # legacy: ALIAS is still a real collection, not an alias yet
    return None


def live_fingerprint(client) -> str | None:
    """Fingerprint of the currently-live index, or None if unknown/unprovisioned.

    A legacy plain collection has no embedded fingerprint, so this returns None
    for it -- which correctly makes the first safe rebuild run (migrating it to
    the alias scheme) instead of being skipped.
    """
    live = resolve_alias(client)
    if live is None:
        return None
    parsed = _parse(live)
    return parsed[0] if parsed else None


# --- build / swap / cleanup ---------------------------------------------------

def create_collection(client, name: str, dim: int):
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def swap_alias(client, new_collection: str):
    """Point the read name at ``new_collection``, atomically where possible.

    Steady state (ALIAS already an alias) is a single atomic delete+create, so
    readers flip over with zero gap. The one exception is the one-time migration
    from a legacy plain collection named ALIAS: we must drop that collection
    before the alias name is free, a brief window in which readers fall back to
    BM25 rather than error out. It happens once, ever.
    """
    aliases = client.get_aliases().aliases
    alias_exists = any(a.alias_name == ALIAS for a in aliases)

    if not alias_exists and client.collection_exists(ALIAS):
        log(f"migrating legacy collection '{ALIAS}' to the alias scheme "
            "(brief BM25-only window for concurrent readers)")
        client.delete_collection(ALIAS)

    ops = []
    if alias_exists:
        ops.append(DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=ALIAS)))
    ops.append(CreateAliasOperation(
        create_alias=CreateAlias(collection_name=new_collection, alias_name=ALIAS)))
    client.update_collection_aliases(change_aliases_operations=ops)
    log(f"alias '{ALIAS}' -> {new_collection}")


def cleanup_orphans(client, keep: str):
    """Delete our old physical collections, keeping the live one.

    Only touches ``<ALIAS>__*`` names, never the live target, and never a
    collection younger than ORPHAN_KEEP_SECONDS -- so a rebuild running
    concurrently in another process can't have its in-progress collection
    swept out from under it.
    """
    now = time.time()
    for coll in client.get_collections().collections:
        name = coll.name
        if name == keep:
            continue
        parsed = _parse(name)
        if not parsed:
            continue
        _, ts = parsed
        if now - ts < ORPHAN_KEEP_SECONDS:
            continue
        try:
            client.delete_collection(name)
            log(f"cleaned up orphan collection {name}")
        except Exception as e:
            log(f"warning: could not delete orphan {name} ({e})")


# --- orchestration ------------------------------------------------------------

def rebuild(client, chunks: list[dict], dim: int, embed_into, *,
            force: bool = False) -> str | None:
    """Safely (re)build the shared index and return the live collection name.

    ``embed_into(collection_name)`` is a callback that embeds every chunk into
    the named (already-created) collection. It is only ever handed a fresh
    throwaway collection, never the live one.

    Returns the physical collection now serving the alias, or None if the build
    was skipped because the index already matches the corpus.
    """
    fp = fingerprint(chunks)

    if not force and live_fingerprint(client) == fp:
        log(f"index already up to date (fingerprint {fp}); nothing to do")
        return resolve_alias(client)

    # Unique name per rebuild (random suffix), so a concurrent rebuild can't be
    # building into this same collection -- no lock needed.
    target = _physical_name(fp)
    try:
        log(f"building fresh collection {target} (dim={dim}); live index untouched")
        create_collection(client, target, dim)
        embed_into(target)
        swap_alias(client, target)
        cleanup_orphans(client, keep=target)
        return target
    except Exception:
        # The alias never moved unless swap_alias succeeded, so the live index is
        # still good. Drop our half-built collection so it doesn't linger.
        try:
            if resolve_alias(client) != target and client.collection_exists(target):
                client.delete_collection(target)
                log(f"rolled back partial collection {target}; live index unchanged")
        except Exception as e:
            log(f"warning: could not clean up partial collection {target} ({e})")
        raise


# --- selfcheck (pure functions only; no Qdrant needed) ------------------------

def _selfcheck():
    a = [{"id": "x", "text": "hello"}, {"id": "y", "text": "world"}]
    b = [{"id": "y", "text": "world"}, {"id": "x", "text": "hello"}]  # reordered
    assert fingerprint(a) == fingerprint(b), "fingerprint must be order-independent"
    assert fingerprint(a) != fingerprint([{"id": "x", "text": "HELLO"}, {"id": "y", "text": "world"}])

    name = _physical_name("abc123def456")
    fp, ts = _parse(name)
    assert fp == "abc123def456" and isinstance(ts, int), (name, fp, ts)

    # two rebuilds in the same second still get distinct names (random suffix)
    assert _physical_name("abc123def456") != _physical_name("abc123def456")

    # legacy pre-random physical names must still parse (the live collection)
    old = f"{ALIAS}{_SEP}abc123def456{_SEP}1700000000"
    assert _parse(old) == ("abc123def456", 1700000000), old

    assert _parse(ALIAS) is None                 # legacy plain name isn't ours
    assert _parse("other__abc__123") is None     # different alias prefix
    print("ok")


if __name__ == "__main__":
    _selfcheck()

"""--incremental must reuse vectors for unchanged text and embed only what's new.

The failure mode this guards is not a crash: handing a chunk the wrong cached
vector produces a confident wrong answer at query time, and reusing a vector some
OTHER model produced would sail straight past Qdrant's dimension check if the dims
happen to match. Both are silent, so both get an assertion here.
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "EGEpipeline"))
import embed_store as es


def _chunk(cid, text):
    return {"id": cid, "file": f"{cid}.cc", "start_line": 1, "end_line": 2,
            "symbol": cid, "language": "cpp", "text": text}


class _LivePoint:
    def __init__(self, text, vector, model):
        self.payload = {"text": text, "embed_model": model}
        self.vector = vector


def _fake_store(monkeypatch, live_points, live="live_coll"):
    """Wire up a fake Qdrant whose live index holds `live_points`.

    Returns (embedded, upserted): texts that reached the embedder, and
    chunk_id -> vector for everything written to the new collection.
    """
    embedded: list[str] = []
    upserted: dict[str, tuple] = {}

    def fake_embed_batch(texts):
        embedded.extend(texts)
        return [[float(len(t)), 0.0] for t in texts]

    class FakeClient:
        def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
            assert collection_name == live, collection_name
            # one page, then stop -- paging itself is Qdrant's, not ours to test
            return (live_points, None) if offset is None else ([], None)

        def upsert(self, collection_name, points):
            for p in points:
                upserted[p.payload["chunk_id"]] = tuple(p.vector)

    monkeypatch.setattr(es, "_embed_batch", fake_embed_batch)
    monkeypatch.setattr(es, "_client", FakeClient())
    monkeypatch.setattr(es.qi, "resolve_alias", lambda _client: live)
    return embedded, upserted


def test_reuses_unchanged_text_and_embeds_only_the_new(monkeypatch):
    chunks = [
        _chunk("a", "old text"),   # already in the live index
        _chunk("b", "brand new"),  # not -- must be embedded
        _chunk("c", "old text"),   # duplicate of a, must reuse the same vector
    ]
    live_points = [_LivePoint("old text", [7.0, 7.0], es.EMBED_MODEL)]
    embedded, upserted = _fake_store(monkeypatch, live_points)

    es.index_into("new_coll", chunks, incremental=True)

    # only the genuinely new text hit the embedder
    assert embedded == ["brand new"], embedded
    # every chunk still landed in the new collection
    assert set(upserted) == {"a", "b", "c"}, sorted(upserted)
    # reused chunks got the LIVE vector, not one of their own or a neighbour's
    assert upserted["a"] == (7.0, 7.0), upserted["a"]
    assert upserted["c"] == (7.0, 7.0), upserted["c"]
    # the new one got its freshly-computed vector
    assert upserted["b"] == (float(len("brand new")), 0.0), upserted["b"]
    print("incremental reuse ✓")


def test_never_reuses_another_models_vector(monkeypatch):
    """Same dim, different model -> Qdrant's dimension check would NOT catch it."""
    chunks = [_chunk("a", "old text")]
    live_points = [_LivePoint("old text", [7.0, 7.0], "some-other-model:latest")]
    embedded, upserted = _fake_store(monkeypatch, live_points)

    es.index_into("new_coll", chunks, incremental=True)

    assert embedded == ["old text"], embedded            # re-embedded, not reused
    assert upserted["a"] == (float(len("old text")), 0.0), upserted["a"]
    print("model-mismatch guard ✓")


def test_default_is_a_full_rebuild(monkeypatch):
    """Without incremental=True the live index is never even consulted."""
    chunks = [_chunk("a", "old text")]
    live_points = [_LivePoint("old text", [7.0, 7.0], es.EMBED_MODEL)]
    embedded, upserted = _fake_store(monkeypatch, live_points)

    def boom(_client):
        raise AssertionError("resolve_alias called on a non-incremental rebuild")
    monkeypatch.setattr(es.qi, "resolve_alias", boom)

    es.index_into("new_coll", chunks)

    assert embedded == ["old text"], embedded
    print("full rebuild stays the default ✓")


def test_no_live_index_falls_back_to_embedding_everything(monkeypatch):
    """First ever build: nothing provisioned, so there's nothing to reuse."""
    chunks = [_chunk("a", "text one"), _chunk("b", "text two")]
    embedded, upserted = _fake_store(monkeypatch, [])
    monkeypatch.setattr(es.qi, "resolve_alias", lambda _client: None)

    es.index_into("new_coll", chunks, incremental=True)

    assert sorted(embedded) == ["text one", "text two"], embedded
    assert set(upserted) == {"a", "b"}, sorted(upserted)
    print("cold-start fallback ✓")

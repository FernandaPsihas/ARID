"""index_into must embed each distinct text once and fan it out to every chunk.

Guards the dedup in embed_store.index_into: 21% of the Tier 1 corpus is duplicate
text, so the win is real, but fanning out wrong would silently give chunks each
other's vectors -- which retrieval would happily serve as a confident wrong answer.
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


def test_dedup_embeds_once_and_fans_out(monkeypatch):
    # "class ParameterSet" three times, "unique A"/"unique B" once each
    chunks = [
        _chunk("a", "class ParameterSet"),
        _chunk("b", "unique A"),
        _chunk("c", "class ParameterSet"),
        _chunk("d", "unique B"),
        _chunk("e", "class ParameterSet"),
    ]

    embedded: list[str] = []

    def fake_embed_batch(texts):
        embedded.extend(texts)
        # deterministic per-text vector so we can verify the fan-out mapping
        return [[float(len(t)), float(sum(map(ord, t)) % 97)] for t in texts]

    upserted = {}

    class FakeClient:
        def upsert(self, collection_name, points):
            for p in points:
                upserted[p.payload["chunk_id"]] = (p.id, tuple(p.vector))

    monkeypatch.setattr(es, "_embed_batch", fake_embed_batch)
    monkeypatch.setattr(es, "_client", FakeClient())

    es.index_into("throwaway", chunks)

    # 5 chunks, 3 distinct texts -> exactly 3 embed inputs, no duplicates
    assert sorted(embedded) == ["class ParameterSet", "unique A", "unique B"], embedded

    # every chunk still got stored, each under its own point id
    assert set(upserted) == {"a", "b", "c", "d", "e"}, sorted(upserted)
    assert len({pid for pid, _ in upserted.values()}) == 5, upserted

    # chunks sharing text share the vector; different text differs
    vec = {cid: v for cid, (_, v) in upserted.items()}
    assert vec["a"] == vec["c"] == vec["e"], vec
    assert vec["b"] != vec["a"] and vec["d"] != vec["a"], vec
    assert vec["b"] != vec["d"], vec

    # the fanned-out vector is the one computed for THAT text, not a neighbour's
    assert vec["a"] == (float(len("class ParameterSet")),
                        float(sum(map(ord, "class ParameterSet")) % 97)), vec["a"]

    print("dedup fan-out ✓")


def test_dedup_keys_on_truncated_text(monkeypatch):
    """Two texts identical up to MAX_EMBED_CHARS but differing after must share a
    vector -- _embed_batch truncates, so embedding both would be pure waste."""
    monkeypatch.setattr(es, "MAX_EMBED_CHARS", 10)
    chunks = [_chunk("a", "x" * 10 + "AAAA"), _chunk("b", "x" * 10 + "BBBB")]

    embedded = []

    def fake_embed_batch(texts):
        embedded.extend(texts)
        return [[1.0, 2.0] for _ in texts]

    class FakeClient:
        def upsert(self, collection_name, points):
            pass

    monkeypatch.setattr(es, "_embed_batch", fake_embed_batch)
    monkeypatch.setattr(es, "_client", FakeClient())

    es.index_into("throwaway", chunks)
    assert embedded == ["x" * 10], embedded
    print("truncation-keyed dedup ✓")

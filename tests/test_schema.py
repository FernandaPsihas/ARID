"""KAN-9 schema checks. Run: python test_schema.py"""

import sys
from dataclasses import fields

from chunk import Chunk, VALID_LANGUAGES, make_chunk_id, validate_chunk


def good():
    return {
        "id": "dunereco_CalibrateHits_42",
        "file": "src/RecoBase/Hit.cc",
        "start_line": 42,
        "end_line": 78,
        "symbol": "CalibrateHits",
        "language": "cpp",
        "text": "void CalibrateHits(...) {}",
    }


assert make_chunk_id("dunereco", "CalibrateHits", 42) == "dunereco_CalibrateHits_42"
assert validate_chunk(good()) == []

bad = good(); bad["language"] = "fhicl"
assert validate_chunk(bad)

bad = good(); bad["end_line"] = 1
assert validate_chunk(bad)

for f in fields(Chunk):
    bad = good(); del bad[f.name]
    assert any(f.name in e for e in validate_chunk(bad)), f.name

sys.stdout.reconfigure(encoding="utf-8")  # ✓ won't encode on Windows cp1252
print("KAN-9 ✓")

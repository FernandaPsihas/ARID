"""Test parse_jsonnet (KAN-8) — real tree-sitter parse."""

import os
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")  # ponytail: Windows console is cp1252, ✓ needs utf-8
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parsers.parse_jsonnet import parse_jsonnet

SOURCE = '''\
local wc = import "wirecell.jsonnet";

{
    lar: {
        DL: 7.2,
        DT: 12.0,
    },
    daq: {
        nticks: 100,
    },
}
'''


def test_parse_jsonnet():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonnet", delete=False) as f:
        f.write(SOURCE)
        path = f.name
    try:
        chunks = parse_jsonnet(path)
    finally:
        os.unlink(path)

    assert all(c["language"] == "jsonnet" for c in chunks)
    by_symbol = {c["symbol"]: c for c in chunks}
    # top-level local bind + each root-object field, real names (not block_N)
    assert "wc" in by_symbol, by_symbol
    assert "lar" in by_symbol, by_symbol
    assert "daq" in by_symbol, by_symbol

    # nested fields stay INSIDE their parent chunk, not split out
    lar = by_symbol["lar"]
    assert "DL" in lar["text"] and "DT" in lar["text"]
    assert "DL" not in by_symbol, "nested field leaked as its own chunk"

    print("KAN-8 ✓")


if __name__ == "__main__":
    test_parse_jsonnet()

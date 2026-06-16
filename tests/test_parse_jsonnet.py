"""Test parse_jsonnet (KAN-8) — fallback brace-depth chunker."""

import os
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")  # ponytail: Windows console is cp1252, ✓ needs utf-8
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parsers.parse_jsonnet import parse_jsonnet

SOURCE = '''\
local greeting = {
    hello: "world",
};

local config = {
    name: "dunereco",
    count: 3,
};
'''


def test_parse_jsonnet():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonnet", delete=False) as f:
        f.write(SOURCE)
        path = f.name
    try:
        chunks = parse_jsonnet(path)
    finally:
        os.unlink(path)

    assert len(chunks) >= 1, chunks
    assert all(c["language"] == "jsonnet" for c in chunks)
    # fallback: blocks should cover their lines contiguously, first block starts at line 1
    assert chunks[0]["start_line"] == 1
    symbols = {c["symbol"] for c in chunks}
    assert "local" in symbols or "greeting" in symbols, symbols

    print("KAN-8 ✓")


if __name__ == "__main__":
    test_parse_jsonnet()

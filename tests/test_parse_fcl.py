"""Test parse_fcl (KAN-5) — shared brace-depth chunker."""

import os
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")  # ponytail: Windows console is cp1252, ✓ needs utf-8
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parsers.parse_fcl import parse_fcl

SOURCE = '''\
#include "services.fcl"

BEGIN_PROLOG
standard_calibration: {
    module_type: Calibration
    threshold: 0.5
}
END_PROLOG

physics: {
    producers: {
        calib: @local::standard_calibration
    }
    path1: [ calib ]
}
'''


def test_parse_fcl():
    with tempfile.NamedTemporaryFile("w", suffix=".fcl", delete=False) as f:
        f.write(SOURCE)
        path = f.name
    try:
        chunks = parse_fcl(path, repo="dunereco")
    finally:
        os.unlink(path)

    by_symbol = {c["symbol"]: c for c in chunks}
    assert "standard_calibration" in by_symbol, by_symbol
    assert "physics" in by_symbol, by_symbol
    assert all(c["language"] == "fcl" for c in chunks)
    # nested producers{} must NOT split physics into pieces — one top-level block
    phys = by_symbol["physics"]
    assert "producers" in phys["text"] and "path1" in phys["text"]

    print("KAN-5 ✓")


if __name__ == "__main__":
    test_parse_fcl()

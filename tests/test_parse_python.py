"""Test parse_python (KAN-6)."""

import os
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")  # ponytail: Windows console is cp1252, ✓ needs utf-8

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parsers.parse_python import parse_python

SOURCE = '''\
def greet(name):
    return f"hi {name}"


class Widget:
    pass
'''


def test_parse_python():
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(SOURCE)
        path = f.name
    try:
        chunks = parse_python(path, repo="dunereco")
    finally:
        os.unlink(path)

    assert len(chunks) == 2, f"expected 2 chunks, got {len(chunks)}"
    by_symbol = {c["symbol"]: c for c in chunks}
    assert "greet" in by_symbol
    assert "Widget" in by_symbol

    greet = by_symbol["greet"]
    assert greet["language"] == "python"
    assert greet["start_line"] == 1
    assert greet["end_line"] == 2
    assert greet["id"] == "dunereco_greet_1"

    widget = by_symbol["Widget"]
    assert widget["start_line"] == 5
    assert widget["end_line"] == 6
    assert "class Widget" in widget["text"]

    print("KAN-6 ✓")


if __name__ == "__main__":
    test_parse_python()

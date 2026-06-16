"""Test parse_cpp (KAN-7)."""

import os
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")  # ponytail: Windows console is cp1252, ✓ needs utf-8
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parsers.parse_cpp import parse_cpp

SOURCE = '''\
int add(int a, int b) {
    return a + b;
}

class Widget {
public:
    void spin();
};

void Widget::spin() {
    return;
}
'''


def test_parse_cpp():
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write(SOURCE)
        path = f.name
    try:
        chunks = parse_cpp(path)
    finally:
        os.unlink(path)

    by_symbol = {c["symbol"]: c for c in chunks}
    assert "add" in by_symbol, by_symbol
    assert "Widget" in by_symbol, by_symbol
    assert "Widget::spin" in by_symbol, by_symbol  # qualified name from nested declarator
    assert all(c["language"] == "cpp" for c in chunks)
    assert by_symbol["add"]["start_line"] == 1

    print("KAN-7 ✓")


if __name__ == "__main__":
    test_parse_cpp()

"""Jsonnet parser for ARID (KAN-8).

FALLBACK: tree-sitter-jsonnet has no working wheel (build fails), so this uses
the shared brace-depth chunker, not a real AST parse. Swap in a grammar when one
is available; the chunk schema stays identical. See parsers/brace_chunker.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parsers.brace_chunker import brace_chunk


def parse_jsonnet(filepath: str) -> list[dict]:
    return brace_chunk(filepath, "jsonnet")

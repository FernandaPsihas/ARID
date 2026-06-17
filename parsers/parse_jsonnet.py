"""Jsonnet parser for ARID (KAN-8).

Real tree-sitter parse via the Sourcegraph grammar. A jsonnet file is one
expression: `local a = ..; local b = ..; { field: .., field: .. }`. The
meaningful chunks are the top-level `local` binds and the direct fields of the
root object — NOT the whole file (which is what the brace chunker produced).

DEP NOTE: the published `tree-sitter-jsonnet` 0.0.1 wheel does not build — its
setup.py omits src/scanner.c. Install from a clone with that one line added.
See requirements.txt.
"""

import os
import sys

from tree_sitter import Language, Parser
import tree_sitter_jsonnet as tsjsonnet

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chunk_schema import make_chunk_id, validate_chunk

# ponytail: tsjsonnet.language() returns an int -> DeprecationWarning on newer
# tree_sitter. Works today; if a tree_sitter bump breaks it, wrap in a PyCapsule.
JSONNET_LANGUAGE = Language(tsjsonnet.language())
_parser = Parser(JSONNET_LANGUAGE)

# structural tokens that aren't the body expression of a node
_STRUCT = {"local", "bind", ";", ",", "{", "}", "=", "comment"}


def _body_expr(node):
    """The expression a document/local_bind wraps (skip binds, punctuation)."""
    body = None
    for c in node.children:
        if c.type not in _STRUCT:
            body = c
    return body


def _name(node, child_type):
    for c in node.children:
        if c.type == child_type:
            return c.text.decode("utf-8")
    return None


def parse_jsonnet(filepath: str) -> list[dict]:
    # --- old brace-chunker fallback (kept for easy revert) ---
    # from parsers.brace_chunker import brace_chunk
    # return brace_chunk(filepath, "jsonnet")
    try:
        with open(filepath, "rb") as f:
            source = f.read()
        tree = _parser.parse(source)
    except Exception as e:
        print(f"warning: failed to parse {filepath}: {e}")
        return []

    # collect (symbol, node) targets: top-level binds + root-object fields
    targets = []
    cur = _body_expr(tree.root_node)
    while cur is not None and cur.type == "local_bind":
        for b in (c for c in cur.children if c.type == "bind"):
            name = _name(b, "id")
            if name:
                targets.append((name, b))
        cur = _body_expr(cur)
    if cur is not None and cur.type == "object":
        for m in (c for c in cur.children if c.type == "member"):
            for fld in (c for c in m.children if c.type == "field"):
                name = _name(fld, "fieldname")
                if name:
                    targets.append((name, fld))

    chunks = []
    for symbol, node in targets:
        start_line = node.start_point[0] + 1
        chunk = {
            "id": make_chunk_id(filepath, symbol, start_line),
            "file": filepath,
            "start_line": start_line,
            "end_line": node.end_point[0] + 1,
            "symbol": symbol,
            "language": "jsonnet",
            "text": node.text.decode("utf-8"),
        }
        errors = validate_chunk(chunk)
        if errors:
            print(f"warning: skipping invalid chunk {symbol}: {errors}")
        else:
            chunks.append(chunk)

    return chunks

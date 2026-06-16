"""Tree-sitter parser for Python files (KAN-6)."""

import os
import sys

from tree_sitter import Language, Parser
import tree_sitter_python as tspython

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chunk_schema import make_chunk_id, validate_chunk

PY_LANGUAGE = Language(tspython.language())
_parser = Parser(PY_LANGUAGE)


def parse_python(filepath: str) -> list[dict]:
    try:
        with open(filepath, "rb") as f:
            source = f.read()
        tree = _parser.parse(source)
    except Exception as e:
        print(f"warning: failed to parse {filepath}: {e}")
        return []

    chunks = []
    # ponytail: iterative DFS over the whole tree; nested defs/classes included on purpose
    # Depth first search my beloved
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in ("function_definition", "class_definition"):
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                symbol = name_node.text.decode("utf-8")
                start_line = node.start_point[0] + 1
                chunk = {
                    "id": make_chunk_id(filepath, symbol, start_line),
                    "file": filepath,
                    "start_line": start_line,
                    "end_line": node.end_point[0] + 1,
                    "symbol": symbol,
                    "language": "python",
                    "text": node.text.decode("utf-8"),
                }
                errors = validate_chunk(chunk)
                if errors:
                    print(f"warning: skipping invalid chunk {symbol}: {errors}")
                else:
                    chunks.append(chunk)
        stack.extend(node.children)

    return chunks

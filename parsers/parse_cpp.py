"""Tree-sitter parser for C++ files (KAN-7)."""

import os
import sys

from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chunk_schema import make_chunk_id, validate_chunk

CPP_LANGUAGE = Language(tscpp.language())
_parser = Parser(CPP_LANGUAGE)


def _func_symbol(node):
    # C++ declarators nest: function_definition -> declarator -> ... -> identifier.
    # Walk down `declarator` field nodes until we hit the name.
    cur = node
    while cur is not None:
        if cur.type in ("identifier", "qualified_identifier", "destructor_name", "operator_name"):
            return cur.text.decode("utf-8")
        nxt = cur.child_by_field_name("declarator")
        if nxt is None:
            # function_declarator holds the name in its declarator child too; if no
            # declarator field, scan children for a name-ish node
            for c in cur.children:
                if c.type in ("identifier", "qualified_identifier", "destructor_name", "operator_name"):
                    return c.text.decode("utf-8")
            return None
        cur = nxt
    return None


def parse_cpp(filepath: str, repo: str = "dunereco") -> list[dict]:
    try:
        with open(filepath, "rb") as f:
            source = f.read()
        tree = _parser.parse(source)
    except Exception as e:
        print(f"warning: failed to parse {filepath}: {e}")
        return []

    chunks = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        symbol = None
        if node.type == "function_definition":
            symbol = _func_symbol(node.child_by_field_name("declarator"))
        elif node.type == "class_specifier":
            name = node.child_by_field_name("name")
            symbol = name.text.decode("utf-8") if name is not None else None

        if symbol:
            start_line = node.start_point[0] + 1
            chunk = {
                "id": make_chunk_id(repo, symbol, start_line),
                "file": filepath,
                "start_line": start_line,
                "end_line": node.end_point[0] + 1,
                "symbol": symbol,
                "language": "cpp",
                "text": node.text.decode("utf-8"),
            }
            errors = validate_chunk(chunk)
            if errors:
                print(f"warning: skipping invalid chunk {symbol}: {errors}")
            else:
                chunks.append(chunk)
        stack.extend(node.children)

    return chunks

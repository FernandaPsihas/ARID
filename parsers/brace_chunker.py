"""Shared brace-depth chunker for languages tree-sitter doesn't cover.

Used by parse_jsonnet and parse_fcl (FHiCL): neither has a working tree-sitter
wheel, and FHiCL is Fermilab-specific so no grammar exists at all. This is NOT
an AST parser — it emits one chunk per top-level `{ }` block by counting braces.
Good enough until a real grammar exists; keep the chunk schema identical to the
tree-sitter parsers so downstream code doesn't care which produced a chunk.

Known ceiling: counts braces inside strings/comments too. A real lexer fixes it.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chunk_schema import make_chunk_id, validate_chunk

_IDENT = re.compile(r"\b[a-zA-Z_]\w*\b")
# leading keywords that aren't the block's real name (jsonnet binds, mostly)
_SKIP = {"local", "function", "assert", "if", "then", "else", "for", "in"}


def _first_ident(line: str) -> str | None:
    for m in _IDENT.finditer(line):
        if m.group(0) not in _SKIP:
            return m.group(0)
    return None


def _symbol(lines: list[str], open_idx: int) -> str:
    """Name for the block whose `{` is on lines[open_idx] (0-indexed)."""
    sym = _first_ident(lines[open_idx])
    if sym:
        return sym
    # FHiCL style puts the name on the line above a lone `{` (`key:\n{`).
    # ponytail: only the nearest preceding non-blank line; a comment/END_PROLOG
    # there yields a junk name, no worse than block_N. Real grammar fixes it.
    j = open_idx - 1
    while j >= 0 and not lines[j].strip():
        j -= 1
    if j >= 0:
        sym = _first_ident(lines[j])
        if sym:
            return sym
    return f"block_{open_idx + 1}"


def brace_chunk(filepath: str, language: str) -> list[dict]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"warning: failed to read {filepath}: {e}")
        return []

    chunks = []
    depth = 0
    buf = []
    start_line = None  # 1-indexed line where the current top-level block opened

    for i, line in enumerate(lines, start=1):
        if depth == 0 and "{" in line:
            start_line = i
        if start_line is not None:
            buf.append(line)
        depth += line.count("{") - line.count("}")
        if start_line is not None and depth <= 0:
            symbol = _symbol(lines, start_line - 1)
            chunk = {
                "id": make_chunk_id(filepath, symbol, start_line),
                "file": filepath,
                "start_line": start_line,
                "end_line": i,
                "symbol": symbol,
                "language": language,
                "text": "".join(buf),
            }
            errors = validate_chunk(chunk)
            if errors:
                print(f"warning: skipping invalid chunk {symbol}: {errors}")
            else:
                chunks.append(chunk)
            depth = 0
            buf = []
            start_line = None

    return chunks

"""FHiCL (.fcl) parser for ARID (KAN-5).

No tree-sitter grammar exists — FHiCL is Fermilab-specific — so this uses the
shared brace-depth chunker, same as parse_jsonnet. FHiCL tables are `key: { ... }`
blocks, which brace-depth chunks cleanly. Top-level lines outside any brace
(#include, BEGIN_PROLOG/END_PROLOG, scalar bindings) are not emitted as chunks.
See parsers/brace_chunker.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parsers.brace_chunker import brace_chunk


def parse_fcl(filepath: str) -> list[dict]:
    return brace_chunk(filepath, "fcl")

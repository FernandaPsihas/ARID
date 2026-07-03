"""FHiCL (.fcl) parser for ARID (KAN-5).

Real recursive-descent parser over a FHiCL token stream, replacing the old
brace-depth heuristic (parsers/brace_chunker.py). Grammar per the FHiCL spec
(mu2e grammar_draft3 + fhicl redmine wiki):

    document   ::= (include | BEGIN_PROLOG | END_PROLOG | definition)*
    definition ::= name [@protect_ignore|@protect_error] ':' value
    name       ::= word ('[' word ']')*          # word may contain dots: a.b.c
    value      ::= table | sequence | atom | reference
    table      ::= '{' (definition | '@table::' name)* '}'
    sequence   ::= '[' [(value | '@sequence::' name) (',' ...)* [',']] ']'
    atom       ::= bare word (number/bool/nil/string) | quoted string
    reference  ::= '@local::' name | '@id::' hash | '@nil' | '@erase'

Comments are `#...` (except `#include`) and `//...`; the lexer drops them, so
braces inside comments/strings can't corrupt chunk boundaries (the old
chunker's known ceiling). One chunk per top-level definition -- table,
sequence, or scalar assignment, prolog contents included; symbol is the full
parameter name. #include lines and prolog markers are parsed, not emitted.
"""

import os
import re
import sys
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chunk_schema import make_chunk_id, validate_chunk


class FclError(Exception):
    pass


Token = namedtuple("Token", "kind text line end_line start end")

_TOKEN_RE = re.compile(
    r"""(?P<WS>[ \t\r\n\f\v]+)
      | (?P<INCLUDE>\#include\b)
      | (?P<COMMENT>\#[^\n]*|//[^\n]*)
      | (?P<STRING>"(?:\\.|[^"\\\n])*"|'[^'\n]*')
      | (?P<DCOLON>::)
      | (?P<PUNCT>[{}\[\],:])
      | (?P<AT>@[A-Za-z_]\w*)
      | (?P<WORD>[A-Za-z0-9_+\-.]+)
    """,
    re.VERBOSE,
)


def _lex(src: str) -> list:
    toks = []
    pos, line = 0, 1
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if m is None:
            raise FclError(f"line {line}: unexpected character {src[pos]!r}")
        kind, text = m.lastgroup, m.group()
        end_line = line + text.count("\n")
        if kind not in ("WS", "COMMENT"):
            if kind in ("PUNCT", "DCOLON"):
                kind = text  # '{' '}' '[' ']' ',' ':' '::' as their own kinds
            toks.append(Token(kind, text, line, end_line, m.start(), m.end()))
        line = end_line
        pos = m.end()
    toks.append(Token("EOF", "", line, line, len(src), len(src)))
    return toks


class _Parser:
    def __init__(self, toks):
        self.toks = toks
        self.i = 0

    def peek(self):
        return self.toks[self.i]

    def next(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, kind):
        t = self.next()
        if t.kind != kind:
            raise FclError(f"line {t.line}: expected {kind!r}, got {t.text!r}")
        return t

    def name(self):
        """Hierarchical name: a.b.c (dots live inside WORD) plus [n] indices."""
        first = self.expect("WORD")
        sym, last = first.text, first
        while self.peek().kind == "[":
            self.next()
            idx = self.expect("WORD")
            last = self.expect("]")
            sym += f"[{idx.text}]"
        return sym, first, last

    def definition(self):
        sym, first, _ = self.name()
        if self.peek().text in ("@protect_ignore", "@protect_error"):
            self.next()
        self.expect(":")
        last = self.value()
        return sym, first, last

    def value(self):
        t = self.peek()
        if t.kind == "{":
            return self.table()
        if t.kind == "[":
            return self.sequence()
        if t.kind == "AT":
            return self.directive()
        if t.kind in ("WORD", "STRING"):
            return self.next()
        raise FclError(f"line {t.line}: expected a value, got {t.text!r}")

    def directive(self):
        t = self.next()
        if t.text in ("@nil", "@erase"):
            return t
        if t.text in ("@local", "@table", "@sequence", "@id"):
            self.expect("::")
            _, _, last = self.name()
            return last
        raise FclError(f"line {t.line}: unknown directive {t.text!r}")

    def table(self):
        self.expect("{")
        while True:
            t = self.peek()
            if t.kind == "}":
                return self.next()
            if t.kind == "EOF":
                raise FclError(f"line {t.line}: unclosed table")
            if t.text == "@table":
                self.directive()
            else:
                self.definition()

    def sequence(self):
        self.expect("[")
        if self.peek().kind == "]":
            return self.next()
        while True:
            if self.peek().text == "@sequence":
                self.directive()
            else:
                self.value()
            t = self.next()
            if t.kind == "]":
                return t
            if t.kind != ",":
                raise FclError(f"line {t.line}: expected ',' or ']' in sequence, got {t.text!r}")
            if self.peek().kind == "]":  # tolerate trailing comma
                return self.next()

    def document(self):
        """Top-level definitions as (symbol, first_token, last_token)."""
        defs = []
        while True:
            t = self.peek()
            if t.kind == "EOF":
                return defs
            if t.kind == "INCLUDE":
                self.next()
                self.expect("STRING")
            elif t.text in ("BEGIN_PROLOG", "END_PROLOG"):
                self.next()
            elif t.text == "@table":  # document-scope splice defines nothing new
                self.directive()
            else:
                defs.append(self.definition())


def _chunks_from_source(src: str, filepath: str) -> list:
    if src.startswith("﻿"):
        src = src[1:]
    chunks = []
    for symbol, first, last in _Parser(_lex(src)).document():
        chunk = {
            "id": make_chunk_id(filepath, symbol, first.line),
            "file": filepath,
            "start_line": first.line,
            "end_line": last.end_line,
            "symbol": symbol,
            "language": "fcl",
            "text": src[first.start:last.end],
        }
        errors = validate_chunk(chunk)
        if errors:
            print(f"warning: skipping invalid chunk {symbol}: {errors}", file=sys.stderr)
        else:
            chunks.append(chunk)
    return chunks


def parse_fcl(filepath: str) -> list:
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            src = f.read()
    except Exception as e:
        print(f"warning: failed to read {filepath}: {e}", file=sys.stderr)
        return []
    try:
        return _chunks_from_source(src, filepath)
    except FclError as e:
        print(f"warning: failed to parse {filepath}: {e}", file=sys.stderr)
        return []


_SELFCHECK_SRC = """\
#include "services.fcl"

BEGIN_PROLOG
standard_calib: {
    module_type: Calibration
    threshold: 0.5
    labels: [ "a{b", plain-atom ]
}
END_PROLOG

process_name: CalibTest

physics: {
    producers: {
        calib: @local::standard_calib
        extra: { @table::standard_calib.sub  gain: 2e-3 }
    }
    path1: [ calib, @sequence::more.paths ]
    debug: @nil
}

physics.producers.calib.threshold @protect_ignore: 0.7
services.msg[0]: true
old_param: @erase
"""


def _selfcheck():
    chunks = _chunks_from_source(_SELFCHECK_SRC, "selfcheck.fcl")
    by = {c["symbol"]: c for c in chunks}
    expected = {
        "standard_calib", "process_name", "physics",
        "physics.producers.calib.threshold", "services.msg[0]", "old_param",
    }
    assert set(by) == expected, f"got {sorted(by)}, expected {sorted(expected)}"
    assert all(c["language"] == "fcl" and not validate_chunk(c) for c in chunks)
    assert by["process_name"]["text"] == "process_name: CalibTest"
    phys = by["physics"]["text"]
    assert "producers" in phys and "path1" in phys and phys.endswith("}")
    print("ok")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "--test":
        _selfcheck()
        sys.exit(0)
    for c in parse_fcl(sys.argv[1]):
        print(f"L{c['start_line']:>4}-{c['end_line']:<4} {c['symbol']}")

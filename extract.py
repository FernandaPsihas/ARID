"""Extract chunks from a repo into chunks.jsonl (KAN-10..13).

Usage:
    python extract.py <repo_root> [subdir ...]

Walks tracked files via `git ls-files`, dispatches each to the parser for its
extension, writes one JSON chunk per line to chunks.jsonl. Chunk `file`/`id`
paths are relative to <repo_root>.

If a parser yields zero chunks for a file, a whole-file fallback chunk is
emitted (symbol "__WHOLE_FILE__") so nothing silently vanishes from retrieval.
These are WARNed per-file, greppable by symbol, and listed in the summary.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chunk_schema import make_chunk_id, validate_chunk
from parsers.parse_python import parse_python
from parsers.parse_cpp import parse_cpp
from parsers.parse_jsonnet import parse_jsonnet
from parsers.parse_fcl import parse_fcl

OUT = "chunks.jsonl"
FALLBACK_SYMBOL = "__WHOLE_FILE__"

# ext -> (parser, language). C++ covers the spread of LArSoft conventions.
_CPP = {".C", ".cc", ".cxx", ".cpp", ".H", ".hh", ".hxx", ".hpp"}
DISPATCH = {
    **{e: (parse_cpp, "cpp") for e in _CPP},
    ".py": (parse_python, "python"),
    ".jsonnet": (parse_jsonnet, "jsonnet"),
    ".fcl": (parse_fcl, "fcl"),
}


def _git_files(root: str, subdirs: list[str]) -> list[str]:
    """Tracked files (repo-relative), optionally restricted to subdirs.

    Filters to files present on disk: a sparse checkout lists tracked paths it
    never wrote to the worktree, and those can't be parsed.
    """
    cmd = ["git", "-C", root, "ls-files", "-z", "--", *subdirs]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [p for p in out.split("\0") if p and os.path.exists(os.path.join(root, p))]


def _fallback_chunk(rel: str, abspath: str, language: str) -> dict | None:
    try:
        with open(abspath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception as e:
        print(f"WARNING: could not read {rel} for fallback: {e}", file=sys.stderr)
        return None
    n = text.count("\n") + 1
    chunk = {
        "id": make_chunk_id(rel, FALLBACK_SYMBOL, 1),
        "file": rel,
        "start_line": 1,
        "end_line": n,
        "symbol": FALLBACK_SYMBOL,
        "language": language,
        "text": text,
    }
    errors = validate_chunk(chunk)
    if errors:
        print(f"WARNING: fallback chunk invalid for {rel}: {errors}", file=sys.stderr)
        return None
    return chunk


def extract(root: str, subdirs: list[str]) -> dict:
    root = os.path.abspath(root)
    # per-language tally: [files, chunks, fallback_files]
    stats = {lang: [0, 0, 0] for _, lang in DISPATCH.values()}
    fallback_files = []

    with open(OUT, "w", encoding="utf-8") as out:
        for rel in _git_files(root, subdirs):
            ext = os.path.splitext(rel)[1]
            entry = DISPATCH.get(ext)
            if entry is None:
                continue
            parser, language = entry
            stats[language][0] += 1
            abspath = os.path.join(root, rel)

            # ponytail: chdir into root so the parser opens the *relative* path and
            # stamps relative file/id for free. Single-threaded, so global cwd is safe;
            # if extraction ever goes parallel, pass abspath + rewrite file/id instead.
            cwd = os.getcwd()
            os.chdir(root)
            try:
                chunks = parser(rel)
            finally:
                os.chdir(cwd)

            if not chunks:
                fb = _fallback_chunk(rel, abspath, language)
                if fb is not None:
                    print(f"WARNING: fallback (whole-file) chunk: {rel}", file=sys.stderr)
                    chunks = [fb]
                    stats[language][2] += 1
                    fallback_files.append(rel)

            for c in chunks:
                out.write(json.dumps(c) + "\n")
            stats[language][1] += len(chunks)

    return {"stats": stats, "fallback_files": fallback_files}


def _print_summary(result: dict) -> None:
    print("\n=== extraction summary ===", file=sys.stderr)
    print(f"{'lang':<9}{'files':>8}{'chunks':>9}{'fallback':>10}", file=sys.stderr)
    for lang, (files, chunks, fb) in sorted(result["stats"].items()):
        print(f"{lang:<9}{files:>8}{chunks:>9}{fb:>10}", file=sys.stderr)
    fbs = result["fallback_files"]
    if fbs:
        print(f"\n{len(fbs)} whole-file fallbacks (grep {FALLBACK_SYMBOL} {OUT}):", file=sys.stderr)
        for p in fbs:
            print(f"  {p}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    result = extract(sys.argv[1], sys.argv[2:])
    _print_summary(result)
    total = sum(s[1] for s in result["stats"].values())
    print(f"\nwrote {total} chunks to {OUT}", file=sys.stderr)

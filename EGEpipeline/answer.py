"""RAG answer step: retrieve fused chunks, ground Claude on them, return the answer.

Retrieval is hybrid (dense Qdrant + BM25, RRF-fused via search_codebase) when the
vector store is up; if Qdrant/Ollama aren't available it falls back to BM25-only
over chunks.jsonl, which is stdlib and always works -- so this runs before the
vector store is provisioned.

Generation goes through the Claude Code CLI as a subprocess (the same way the
ARID/Jarvis Discord bots call Claude in assistant_core.py), NOT the Anthropic
SDK. Auth comes from the CLI's own login, so no API key is needed. Override the
exe with the ARID_CLAUDE_EXE env var.

Usage:
    python EGEpipeline/answer.py "how is neutrino energy reconstructed?"
"""

import json
import os
import subprocess
import sys

sys.path.append(os.path.dirname(__file__))

CLAUDE_EXE = os.environ.get(
    "ARID_CLAUDE_EXE", r"C:\Users\Cephandrius\.local\bin\claude.exe"
)
GEN_MODEL = "opus"     # --model alias the CLI understands
GEN_TIMEOUT = 240      # seconds for the claude -p call
TOP_K = 6              # chunks fed as context
SNIPPET = 2000         # max chars shown per chunk (big funcs get truncated in the prompt)

SYSTEM = (
    "You are a code assistant for the DUNE dunereco (LArSoft) codebase. "
    "Answer the question using ONLY the code snippets provided. Cite the file and "
    "line range for each claim, e.g. (FDSensOpt/NeutrinoEnergyRecoAlg.cc L146-178). "
    "If the snippets don't contain the answer, say so plainly -- do not invent code."
)

_BM25 = None  # cached BM25 index for the fallback path


def _format_context(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        head = f"### {c['file']} L{c['start_line']}-{c['end_line']}  ({c['symbol']})"
        blocks.append(f"{head}\n```{c['language']}\n{c['text'][:SNIPPET]}\n```")
    return "\n\n".join(blocks)


def _retrieve(query: str, top_k: int) -> list[dict]:
    """Hybrid retrieval, falling back to BM25-only if dense search is unavailable."""
    try:
        from search import search_codebase  # lazy: pulls qdrant/ollama/bm25 only when run
        return search_codebase(query, top_k=top_k)
    except Exception as e:
        # Qdrant/Ollama not up (or qdrant-client/ollama not installed yet) --
        # BM25-only still gives useful results and needs nothing but chunks.jsonl.
        print(f"[ask] dense retrieval unavailable ({type(e).__name__}: {e}); "
              "using BM25-only.", file=sys.stderr)
        from search import CHUNKS_PATH
        from search_bm25 import BM25, load_chunks
        global _BM25
        if _BM25 is None:
            _BM25 = BM25(load_chunks(CHUNKS_PATH))
        return _BM25.search(query, top_k=top_k)


def _generate(query: str, chunks: list[dict]) -> str:
    """Ground Claude on the retrieved snippets via the Claude Code CLI."""
    prompt = f"Question: {query}\n\nCode snippets:\n\n{_format_context(chunks)}"
    args = [CLAUDE_EXE, "-p", prompt, "--model", GEN_MODEL, "--tools", "",
            "--append-system-prompt", SYSTEM, "--output-format", "json"]
    env = dict(os.environ)
    env["PONYTAIL_DEFAULT_MODE"] = "off"
    # Windows: invisible console so claude.exe doesn't flash a window.
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    res = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=GEN_TIMEOUT,
                         stdin=subprocess.DEVNULL, creationflags=creationflags)
    raw = res.stdout or ""
    i = raw.find('{"type":"result"')
    if i == -1:
        return "(generation failed) " + ((res.stderr or "").strip()[:300] or "no output")
    obj = json.JSONDecoder().raw_decode(raw[i:])[0]
    return (obj.get("result") or "(no reply)").strip()


def answer(query: str, top_k: int = TOP_K) -> str:
    """Retrieve, ground Claude, and return the answer plus its sources."""
    chunks = _retrieve(query, top_k=top_k)
    if not chunks:
        return "No relevant chunks found."
    body = _generate(query, chunks)
    sources = "\n".join(
        f"  {c['file']}  L{c['start_line']}-{c['end_line']}  {c['symbol']}"
        for c in chunks
    )
    return f"{body}\n\nSources:\n{sources}"


def _selfcheck():
    ctx = _format_context([
        {"file": "a.cc", "start_line": 1, "end_line": 9, "symbol": "Foo",
         "language": "cpp", "text": "x" * 5000},
    ])
    assert "a.cc L1-9  (Foo)" in ctx
    assert "```cpp" in ctx
    assert ctx.count("x") == SNIPPET, "snippet not truncated"  # big chunk capped
    print("ok")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "--test":
        _selfcheck()
        sys.exit(0)
    print(answer(" ".join(sys.argv[1:])))

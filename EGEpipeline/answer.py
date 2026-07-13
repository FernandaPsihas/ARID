"""RAG answer step: retrieve fused chunks, ground a local model on them, return the answer.

Retrieval is hybrid (dense Qdrant + BM25, RRF-fused via search_codebase) when the
vector store is up; if Qdrant/Ollama aren't available it falls back to BM25-only
over chunks.jsonl, which is stdlib and always works -- so this runs before the
vector store is provisioned.

Generation runs entirely locally through Ollama (qwen3-coder:30b, a 30B-total /
3B-active MoE -- fast on limited VRAM and leaves headroom on the GPU for the
qwen3-embedding:0.6b embedding model to run alongside it). No API key or Claude
CLI login needed. `arid_mcp.py` imports `_generate` from here so the MCP server
and this CLI script always use the same model and prompt.

Usage:
    python EGEpipeline/answer.py "how is neutrino energy reconstructed?"
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

GEN_MODEL = "qwen3-coder:30b"  # Ollama tag; pull with `ollama pull qwen3-coder:30b`
NUM_CTX = 8192          # TOP_K=6 chunks * SNIPPET=2000 chars is ~4-5k tokens of context, so
                        # this comfortably covers it. Measured: 32768 pushed Ollama's KV cache
                        # to ~44GB (nearly the whole 48GB card), leaving no room for the
                        # embedding model; 8192 keeps real headroom (`ollama ps` to check).
TOP_K = 6               # chunks fed as context
SNIPPET = 2000          # max chars shown per chunk (big funcs get truncated in the prompt)

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


def _generate(query: str, chunks: list[dict], stream: bool = False) -> str:
    """Ground the local model on the retrieved snippets via Ollama.

    stream=True prints tokens to stdout as they arrive (for interactive CLI use);
    either way the full response text is returned. arid_mcp.py calls this with the
    default stream=False since an MCP tool call returns one string, not a live feed.
    """
    import ollama  # lazy: keeps BM25-only / self-check paths dependency-free
    prompt = f"Question: {query}\n\nCode snippets:\n\n{_format_context(chunks)}"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]

    if not stream:
        resp = ollama.chat(model=GEN_MODEL, messages=messages, options={"num_ctx": NUM_CTX})
        return (resp["message"]["content"] or "(no reply)").strip()

    parts = []
    for chunk in ollama.chat(model=GEN_MODEL, messages=messages,
                              options={"num_ctx": NUM_CTX}, stream=True):
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        parts.append(token)
    print()
    return "".join(parts).strip()


def answer(query: str, top_k: int = TOP_K, stream: bool = False) -> str:
    """Retrieve, ground the local model, and return the answer plus its sources.

    If stream=True, the answer body is also printed live to stdout as it generates.
    """
    chunks = _retrieve(query, top_k=top_k)
    if not chunks:
        return "No relevant chunks found."
    body = _generate(query, chunks, stream=stream)
    sources = "\n".join(
        f"  {c['file']}  L{c['start_line']}-{c['end_line']}  {c['symbol']}"
        for c in chunks
    )
    if stream:
        print(f"\nSources:\n{sources}")
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
    answer(" ".join(sys.argv[1:]), stream=True)

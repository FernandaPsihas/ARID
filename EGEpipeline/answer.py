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

# Guarantee we're running under the repo .venv with deps installed (building it
# on first run if needed) BEFORE importing anything third-party. Shared across
# the entry points via env_setup; may re-exec the process, so keep it first.
sys.path.insert(0, os.path.dirname(__file__))
from env_setup import ensure_env
ensure_env()

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

class GenerationUnavailable(RuntimeError):
    """The local generation model couldn't be reached or run.

    Retrieval can still have succeeded -- this only covers the generate step, so
    callers catch it and surface the retrieved sources with a clear message
    instead of dumping a raw traceback (e.g. wrong Python env, or the shared
    Ollama being down)."""


def _format_context(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        head = f"### {c['file']} L{c['start_line']}-{c['end_line']}  ({c['symbol']})"
        blocks.append(f"{head}\n```{c['language']}\n{c['text'][:SNIPPET]}\n```")
    return "\n\n".join(blocks)


def _retrieve(query: str, top_k: int) -> list[dict]:
    """Hybrid retrieval. search_codebase degrades on its own -- BM25-only if the
    dense side (Qdrant/Ollama) is down, dense-only if there's no local chunks.jsonl,
    and [] with a warning only if both halves are unavailable."""
    from search import search_codebase  # lazy: pulls qdrant/ollama/bm25 only when run
    return search_codebase(query, top_k=top_k)


def _generate(query: str, chunks: list[dict], stream: bool = False,
               history: list[dict] | None = None, num_ctx: int | None = None) -> str:
    """Ground the local model on the retrieved snippets via Ollama.

    stream=True prints tokens to stdout as they arrive (for interactive CLI use);
    either way the full response text is returned. arid_mcp.py calls this with the
    default stream=False since an MCP tool call returns one string, not a live feed.

    history, if given, is a list of prior {"role": "user"/"assistant", "content": ...}
    turns (plain question/answer text, no code chunks -- chat.py uses this for
    multi-turn sessions so old turns don't re-inflate the prompt with stale snippets).
    """
    try:
        import ollama  # lazy: keeps BM25-only / self-check paths dependency-free
    except ModuleNotFoundError as e:
        raise GenerationUnavailable(
            "the `ollama` package isn't installed in this Python environment — "
            "activate the project venv (`source .venv/bin/activate`) or run inside "
            "the Docker app container."
        ) from e

    prompt = f"Question: {query}\n\nCode snippets:\n\n{_format_context(chunks)}"
    messages = [{"role": "system", "content": SYSTEM}, *(history or []),
                {"role": "user", "content": prompt}]
    ctx = num_ctx or NUM_CTX

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        if not stream:
            resp = ollama.chat(model=GEN_MODEL, messages=messages, options={"num_ctx": ctx})
            return (resp["message"]["content"] or "(no reply)").strip()

        parts = []
        for chunk in ollama.chat(model=GEN_MODEL, messages=messages,
                                  options={"num_ctx": ctx}, stream=True):
            token = chunk["message"]["content"]
            print(token, end="", flush=True)
            parts.append(token)
        print()
        return "".join(parts).strip()
    except Exception as e:
        # Down service, model not pulled, connection refused, etc. -> a clear,
        # actionable error instead of a raw traceback out of the CLI/chat loop.
        raise GenerationUnavailable(
            f"couldn't reach the Ollama generation service at {host} "
            f"({type(e).__name__}: {e}). Check it's running and OLLAMA_HOST is right; "
            f"the model '{GEN_MODEL}' must be pulled there."
        ) from e


def answer(query: str, top_k: int = TOP_K, stream: bool = False) -> str:
    """Retrieve, ground the local model, and return the answer plus its sources.

    If stream=True, the answer body is also printed live to stdout as it generates.
    """
    chunks = _retrieve(query, top_k=top_k)
    if not chunks:
        return "No relevant chunks found."
    sources = "\n".join(
        f"  {c['file']}  L{c['start_line']}-{c['end_line']}  {c['symbol']}"
        for c in chunks
    )
    try:
        body = _generate(query, chunks, stream=stream)
    except GenerationUnavailable as e:
        # Retrieval worked; only generation is down. Still return the sources so
        # the researcher gets the relevant files even without a written answer.
        body = f"[generation unavailable] {e}"
        if stream:
            print(body)
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

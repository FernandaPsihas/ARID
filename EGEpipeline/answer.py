"""RAG answer step: retrieve fused chunks, ground qwen2.5-coder on them, stream answer.

Usage:
    python EGEpipeline/answer.py "how is neutrino energy reconstructed?"
"""

import os
import sys

import ollama

sys.path.append(os.path.dirname(__file__))

GEN_MODEL = "qwen2.5-coder:7b"
TOP_K     = 6      # chunks fed as context
SNIPPET   = 2000   # max chars shown per chunk (big funcs get truncated in the prompt)
NUM_CTX   = 8192   # qwen2.5-coder defaults to 2048 in ollama — too small for RAG context

SYSTEM = (
    "You are a code assistant for the DUNE dunereco (LArSoft) codebase. "
    "Answer the question using ONLY the code snippets provided. Cite the file and "
    "line range for each claim, e.g. (FDSensOpt/NeutrinoEnergyRecoAlg.cc L146-178). "
    "If the snippets don't contain the answer, say so plainly — do not invent code."
)


def _format_context(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        head = f"### {c['file']} L{c['start_line']}-{c['end_line']}  ({c['symbol']})"
        blocks.append(f"{head}\n```{c['language']}\n{c['text'][:SNIPPET]}\n```")
    return "\n\n".join(blocks)


def answer(query: str, top_k: int = TOP_K) -> None:
    from search import search_codebase  # lazy: pulls qdrant/ollama/bm25 only when run
    sys.stdout.reconfigure(encoding="utf-8")  # model emits smart quotes/em-dashes; cp1252 console mangles them

    chunks = search_codebase(query, top_k=top_k)
    if not chunks:
        print("No relevant chunks found.")
        return

    prompt = f"Question: {query}\n\nCode snippets:\n\n{_format_context(chunks)}"
    stream = ollama.chat(
        model=GEN_MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}],
        options={"num_ctx": NUM_CTX},
        stream=True,
    )
    for part in stream:
        print(part["message"]["content"], end="", flush=True)
    print("\n\nSources:")
    for c in chunks:
        print(f"  {c['file']}  L{c['start_line']}-{c['end_line']}  {c['symbol']}")


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
    answer(" ".join(sys.argv[1:]))

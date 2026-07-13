"""ARID MCP server — exposes the dunereco retrieval pipeline as tools, nothing else.

A remote Claude reaches this over `ssh <user>@<tailscale-ip>`, where the SSH key's
forced command launches THIS script. The SSH session's stdin/stdout become the MCP
stdio channel, so the only thing a connecting client can do is call these tools —
no shell, no arbitrary commands. Auth = SSH key; encryption = SSH over Tailscale.

Run locally for a smoke test:  python EGEpipeline/arid_mcp.py --test
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))  # import sibling search.py / answer.py

from mcp.server.fastmcp import FastMCP

from answer import _generate
from search import search_codebase

mcp = FastMCP("arid")


@mcp.tool()
def search_arid(query: str, top_k: int = 10) -> list[dict]:
    """Hybrid (dense + BM25) search over the DUNE dunereco (LArSoft) codebase.

    Returns the top_k most relevant code chunks, best first, each with file,
    line range, symbol, language, the source text, and a fusion score. Use this
    to retrieve grounding context and write the answer yourself.
    """
    return search_codebase(query, top_k=top_k)


@mcp.tool()
def ask_arid(query: str, top_k: int = 6) -> str:
    """Retrieve from dunereco and return a grounded, cited answer (local qwen3-coder:30b).

    Convenience tool: does the same retrieval as search_arid, then has the local
    model write a cited answer. Prefer search_arid if you want to compose the
    answer yourself from raw chunks.
    """
    chunks = search_codebase(query, top_k=top_k)
    if not chunks:
        return "No relevant chunks found in the dunereco codebase."
    body = _generate(query, chunks)
    src = "\n".join(f"  {c['file']}  L{c['start_line']}-{c['end_line']}  {c['symbol']}" for c in chunks)
    return f"{body}\n\nSources:\n{src}"


def _warmup() -> None:
    # ponytail: the first search_codebase lazily imports embed_store, builds the BM25 index,
    # and opens the qdrant/ollama clients — ~tens of seconds inside the MCP event loop on the
    # FIRST request. Doing it here at startup keeps every actual tool call ~1s.
    try:
        search_codebase("warmup", top_k=1)
    except Exception as e:
        print(f"warmup: dense search unavailable ({e}); BM25-only until services return", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # smoke test: tools registered + retrieval returns something (BM25 works even with services down)
        import asyncio
        names = {t.name for t in asyncio.run(mcp.list_tools())}
        assert names == {"search_arid", "ask_arid"}, names
        hits = search_arid("neutrino energy reconstruction", top_k=3)
        assert hits and "file" in hits[0], hits
        print("ok:", names, "->", len(hits), "hits")
        sys.exit(0)
    _warmup()
    mcp.run()  # stdio transport

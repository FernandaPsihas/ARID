# ARID — Augmented Research Integration for Dune

ARID is a RAG (retrieval-augmented generation) pipeline that lets you ask natural-language questions about the [dunereco](https://github.com/DUNE/dunereco) LArSoft codebase and get grounded, citation-backed answers.

**Example:**
```
python EGEpipeline/answer.py "how is neutrino energy reconstructed?"
```
Returns an answer, generated entirely locally via Ollama, citing the exact files and line ranges it drew from.

---

## Architecture

```
dunereco repo
     │
     ▼
extract.py          Walk git-tracked files, parse into chunks
     │              (C++ / Python / FHiCL / Jsonnet parsers in parsers/)
     ▼
chunks.jsonl        One chunk per line: {id, file, start/end line, symbol, language, text}
     │
     ├──► BM25 index      (search_bm25.py — stdlib, always available)
     │
     └──► Qdrant + Ollama (embed_store.py — qwen3-embedding:0.6b via Ollama)
               │
               └──► RRF fusion (search.py — combines dense + BM25, top-k)
                         │
                         ▼
                    answer.py       Retrieve chunks → ground qwen3-coder:30b (Ollama) → answer + sources
```

Hybrid retrieval falls back to BM25-only if Qdrant/Ollama aren't running — the pipeline is useful without the vector store.

Generation runs locally too: `answer.py` and the MCP server (`arid_mcp.py`) both call
`qwen3-coder:30b` via Ollama — a 30B-total / 3B-active MoE, so it's fast even without
a huge GPU, and its ~19GB (Q4_K_M) footprint leaves plenty of VRAM headroom for the
embedding model to run alongside it. No API key or separate CLI login needed.

---

## Chunk Schema

Every chunk in `chunks.jsonl` follows this schema (defined in `chunk_schema.py`):

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | `"{file}_{symbol}_{start_line}"` — unique across the repo |
| `file` | str | Repo-relative path |
| `start_line` | int | First line of the symbol |
| `end_line` | int | Last line of the symbol |
| `symbol` | str | Function/class/module name; `__WHOLE_FILE__` for fallback chunks |
| `language` | str | `cpp`, `python`, `fcl`, or `jsonnet` |
| `text` | str | Raw source text of the chunk |

---

## Setup (containerized, one command)

```
docker compose up -d --build
```

That's the whole install, on any machine with Docker — no GPU or CUDA required.
It builds the pipeline image, starts Qdrant + Ollama (CPU), clones `dunereco`,
extracts chunks, and embeds + indexes them into Qdrant — `docker/bootstrap.py`
runs all of that automatically inside the `app` container. Every step is
idempotent (it checks what's already on disk / in Qdrant first), so re-running
`docker compose up` after the first success is a few-second no-op — safe to use
as a "resume" command too. Follow progress with `docker compose logs -f app`;
first run takes a while (repo clone + embedding — CPU embedding of the full
dunereco corpus is tens of minutes), later runs are instant.

**Sharing a host with other researchers?** Several people can each run their own
ARID stack on the same machine at once — you just need a unique project name and
distinct host ports so you don't collide on `arid-*` container names or the
`6333`/`11434` host ports (`address already in use`). Copy the template and edit it:
```
cp .env.example .env      # set COMPOSE_PROJECT_NAME + QDRANT_PORT/OLLAMA_PORT to unique values
```
Compose reads `.env` automatically, so `docker compose up -d --build` then brings
up *your* isolated stack. Nothing inside the pipeline changes — the container-internal
ports and service addresses (`http://ollama:11434`, `http://qdrant:6333`) stay fixed;
only the host-side bindings move. Each project name gets its own volumes, so you
pull the models / embed the corpus once for yourself.

**Have an NVIDIA GPU?** Add the GPU override for much faster embedding:
```
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```
Requires the host to have the NVIDIA driver and
[`nvidia-container-toolkit`](https://github.com/NVIDIA/nvidia-container-toolkit)
installed and registered with Docker (`docker info` should list an `nvidia`
runtime) — that part can't be containerized, it's kernel/driver level. Without
it, leave the GPU file out entirely; the base `docker-compose.yml` never
requests a GPU device, so it can't fail to find one.

**Query it:**
```
docker compose exec app python EGEpipeline/search_bm25.py chunks.jsonl "APAChannelsIntersect"
docker compose exec app python EGEpipeline/search.py "electron lifetime calibration"
docker compose exec app python EGEpipeline/answer.py "how is neutrino energy reconstructed?"
```
`answer.py`'s generation step runs against the containerized Ollama, so it works
out of the box — no separate login or API key needed.

**Interactive multi-turn CLI for researchers:**
```
docker compose exec -it app python EGEpipeline/chat.py
```
(or, from the host, against the containerized Qdrant/Ollama published on
`localhost`: `source .venv/bin/activate && python EGEpipeline/chat.py`)

Asks for a name, then takes as many questions as you want in one sitting —
answers stream token-by-token and later questions can reference earlier
answers (e.g. "what about the NC case?"). After each answer you rate it
`good` / `needs improvement` / `skip`; "needs improvement" asks for one or
more reasons (missing info, wrong citations, hallucinated, too vague,
retrieval missed the code) plus optional free-text notes. The whole session
is saved after every turn to `feedback/<researcher>/<timestamp>.json` — safe
against a crash or Ctrl+C mid-session. Note: multi-turn memory only affects
*generation* (the model sees prior Q&A text) — *retrieval* still searches
using just the literal follow-up text, so a heavily pronoun-dependent
follow-up ("how is it configured?") can retrieve weaker chunks than a
self-contained one ("how is CalorimetryAlg configured?").

Useful env vars for `docker compose up` (set in your shell or a `.env` file):
| Var | Default | Purpose |
|-----|---------|---------|
| `DUNERECO_URL` | `github.com/DUNE/dunereco.git` | corpus to clone |
| `DUNERECO_REF` | repo default branch | branch/tag to check out |
| `FORCE_EXTRACT` | unset | `1` re-extracts `chunks.jsonl` even if present |
| `FORCE_EMBED` | unset | `1` re-embeds even if Qdrant already has points |

### Manual / non-Docker setup

Still works if you'd rather run things directly:
```
pip install -r requirements.txt          # needs a C compiler (tree-sitter grammars)
ollama pull qwen3-embedding:0.6b         # requires Ollama running locally
ollama pull qwen3-coder:30b              # generation model used by answer.py / arid_mcp.py
docker run -p 6333:6333 qdrant/qdrant    # requires Qdrant running locally
git clone https://github.com/DUNE/dunereco.git
python extract.py dunereco/                        # writes chunks.jsonl
python EGEpipeline/embed_store.py chunks.jsonl      # embed + index
python EGEpipeline/answer.py "your question"
```
BM25-only mode needs neither Ollama nor Qdrant — just `pip install` and skip straight to `search_bm25.py`.

---

## File Structure

```
ARID/
├── extract.py              Chunk extraction entry point
├── chunk_schema.py         Schema definition + validation
├── chunks.jsonl            Output of extract.py (gitignored if large)
├── requirements.txt
├── parsers/
│   ├── parse_cpp.py        C++ parser (tree-sitter)
│   ├── parse_python.py     Python parser (tree-sitter)
│   ├── parse_jsonnet.py    Jsonnet parser (tree-sitter)
│   ├── parse_fcl.py        FHiCL parser (in-house tree-sitter)
│   └── brace_chunker.py    Shared brace-depth chunking utility
├── EGEpipeline/
│   ├── embed_store.py      Embedding + Qdrant indexing + dense search
│   ├── search_bm25.py      BM25 keyword search
│   ├── search.py           Hybrid RRF fusion (dense + BM25)
│   ├── answer.py           RAG answer step (retrieve → generate → cite)
│   ├── chat.py             Interactive multi-turn CLI with researcher rating + logging
│   └── arid_mcp.py         MCP server exposing search_arid/ask_arid over SSH stdio
├── feedback/                gitignored: chat.py session logs, one JSON per researcher/session
├── eval/
│   └── gold_queries.json   Gold query set for retrieval eval (KAN-24)
├── docker/
│   └── bootstrap.py        Container entrypoint: clone → extract → embed+index, idempotent
├── docker-compose.yml       qdrant + ollama (CPU) + app, one-command install
├── docker-compose.gpu.yml  optional override: adds NVIDIA GPU passthrough to ollama
├── Dockerfile
└── dunereco/               Clone of the target repo (gitignored, not baked into the image)
```

---

## Evaluation

`eval/gold_queries.json` contains a starter set of 7 natural-language queries with expected chunk targets for measuring retrieval quality (Recall@k, MRR). All queries are currently `unverified-name-match` status — targets were derived by pattern-matching symbol names and need confirmation against actual function bodies (KAN-24).

---

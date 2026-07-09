# ARID — Augmented Research Integration for Dune

ARID is a RAG (retrieval-augmented generation) pipeline that lets you ask natural-language questions about the [dunereco](https://github.com/DUNE/dunereco) LArSoft codebase and get grounded, citation-backed answers.

**Example:**
```
python EGEpipeline/answer.py "how is neutrino energy reconstructed?"
```
Returns a Claude-generated answer citing the exact files and line ranges it drew from.

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
                    answer.py       Retrieve chunks → ground Claude → return answer + sources
```

Hybrid retrieval falls back to BM25-only if Qdrant/Ollama aren't running — the pipeline is useful without the vector store.

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
```

`answer.py`'s generation step shells out to the Claude Code CLI, which needs
its own login — that's user-specific and isn't baked into the image. Either
install and `claude login` inside the container yourself, or run `answer.py`
from the host against the containerized Qdrant/Ollama (ports `6333`/`11434`
are published to `localhost`):
```
pip install -r requirements.txt   # host-side, only needed for this path
ARID_CLAUDE_EXE=claude python EGEpipeline/answer.py "how is neutrino energy reconstructed?"
```

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
│   └── answer.py           RAG answer step (retrieve → generate → cite)
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

# ARID — AI-Assisted Repository Interrogation for DUNE

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

## Setup

**Dependencies:**
```
pip install -r requirements.txt
```

Requires [Ollama](https://ollama.com) running locally with the embedding model pulled:
```
ollama pull qwen3-embedding:0.6b
```

And [Qdrant](https://qdrant.tech) running locally (default `localhost:6333`):
```
docker run -p 6333:6333 qdrant/qdrant
```

BM25-only mode works without either — just `pip install` the requirements and skip Ollama/Qdrant.

---

## Usage

**Step 1 — Extract chunks from the repo:**
```
python extract.py dunereco/
```
Writes `chunks.jsonl` to the project root.

**Step 2 — Embed and index (requires Ollama + Qdrant):**
```
python EGEpipeline/embed_store.py chunks.jsonl
```

**Step 3 — Ask a question:**
```
python EGEpipeline/answer.py "how does the CC neutrino selection pick the best shower?"
```

**Search only (no generation):**
```
python EGEpipeline/search.py "electron lifetime calibration"
python EGEpipeline/search_bm25.py chunks.jsonl "APAChannelsIntersect"
```

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
└── dunereco/               Submodule / clone of the target repo
```

---

## Evaluation

`eval/gold_queries.json` contains a starter set of 7 natural-language queries with expected chunk targets for measuring retrieval quality (Recall@k, MRR). All queries are currently `unverified-name-match` status — targets were derived by pattern-matching symbol names and need confirmation against actual function bodies (KAN-24).

---

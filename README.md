# ARID — Augmented Research Integration for Dune

ARID is a RAG (retrieval-augmented generation) pipeline that lets you ask natural-language questions about the [dunereco](https://github.com/DUNE/dunereco) LArSoft codebase and get grounded, citation-backed answers.

**Example:**
```
python EGEpipeline/answer.py "how is neutrino energy reconstructed?"
```
Returns an answer, generated entirely locally via Ollama, citing the exact files and line ranges it drew from.

---

## Quickstart (for researchers on a shared host)

The Qdrant + Ollama services already run **once** for everyone on the shared
machine — you don't start them, download models, or embed anything. From your own
clone of this repo, just clone and ask.

**Clone, then ask — that's the whole thing:**
```
git clone <this repo> ARID && cd ARID
python3 EGEpipeline/chat.py                                  # interactive multi-turn
python3 EGEpipeline/answer.py "how are pixel maps built for CVN?"   # one-shot
```
The **first** run bootstraps its own environment — builds the venv, installs the
query dependencies — then continues into the tool. It's a one-time ~30s step
you'll see labelled `ARID: setting up the query environment`; every run after is
instant. No `source .venv/bin/activate`, no Docker, no models to download, no
admin rights, no C compiler. The scripts default to the shared services on
`localhost`, so there's nothing to configure.

Prefer to do that setup explicitly (say, to pre-build before a demo instead of on
first question)? Run `./setup.sh` — it does exactly what the first-run bootstrap
does, is idempotent, and then `chat.py` starts straight up.

Under the hood, none of which you need to care about if it just works:
- **The odd venv bootstrap.** A plain `python3 -m venv .venv` fails on hosts where
  the `python3-venv` package isn't installed (`ensurepip is not available`) — which
  includes QuarkyLab. The script detects that and falls back to a `--without-pip`
  venv plus `get-pip.py`, so it works either way.
- **`requirements-query.txt`**, not `requirements.txt` — the full file builds the
  tree-sitter grammars from source and needs a C toolchain. That's only for
  `extract.py`, which you don't run.
- **No `sync_chunks.py` step** — the first query auto-builds
  `chunks.jsonl` (the BM25 half of retrieval) by pulling chunk text back out of the
  shared Qdrant. If Qdrant is unreachable at that moment it just runs dense-only
  with a warning, and retries on the next query. You can still run
  `.venv/bin/python EGEpipeline/sync_chunks.py` by hand to pre-build it if you'd
  rather not pay that one-time cost on your first question.

> **Note:** the scripts re-exec into `.venv` on their own, so running the system
> `python3` no longer gives `ModuleNotFoundError: No module named 'ollama'` — it
> just works. (This only applies to `chat.py`/`answer.py`; for ad-hoc `python`
> commands against the pipeline, activate the venv or call `.venv/bin/python`.)

Full setup, the shared-services model, and admin/provisioning steps are below.

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

Retrieval degrades in **both** directions rather than crashing: BM25-only if
Qdrant/Ollama are unreachable, dense-only if there's no local `chunks.jsonl`.
It returns nothing (with a warning) only if both halves are unavailable.

Generation runs locally too: `answer.py` and the MCP server (`arid_mcp.py`) both call
`qwen3-coder:30b` via Ollama — a 30B-total / 3B-active MoE, so it's fast even without
a huge GPU. The weights are ~18 GB on disk (Q4_K_M); resident cost is ~25 GB with
the KV cache at `NUM_CTX=8192`, plus ~3 GB for the embedding model — about 28 GB of
the RTX 8000's 46 GB, leaving real headroom. No API key or separate CLI login needed.

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

## Setup — shared services, many researchers

ARID is built for a shared machine: the heavy services (Qdrant + Ollama) run
**once** for everyone, and each researcher keeps their own clone of this repo and
just **queries** those shared services. That way the models load into VRAM once
(not once per person) and the corpus is embedded once — see
[Shared vs. isolated](#shared-vs-isolated-why-this-layout) for the reasoning.

### 1. Stand up the shared services (once per host)

```
./services.sh
```
That's it. `services.sh` is **detect-or-host** and idempotent:

- If a Qdrant (6333) and Ollama (11434) are **already running**, it reuses them as
  the shared instance and starts nothing.
- If either is missing, it **hosts** the shared stack (`docker-compose.shared.yml`),
  automatically adding the NVIDIA GPU override when the host has a usable GPU — so
  `up` is never accidentally CPU-only (see the GPU note below for why that matters).
- If the `dunereco` corpus isn't indexed yet, it **provisions** it (clone + extract
  + embed); if it's already there, it skips.

Under the hood it runs the plain compose commands, which you can also invoke directly:
```
docker compose -f docker-compose.shared.yml -f docker-compose.gpu.yml up -d qdrant ollama
docker compose -f docker-compose.shared.yml -f docker-compose.gpu.yml run --rm provision
```
(drop `-f docker-compose.gpu.yml` on a host with no GPU.)

Models and the embedded corpus persist in the standard compose-owned named
volumes `arid-shared_ollama_data` and `arid-shared_qdrant_data`, so a restart
never re-pulls or re-embeds. `services.sh` reuses an already-running stack rather
than re-`up`-ing it, so a stray `up` can't shadow those volumes.

`provision` (`docker/bootstrap.py`) pulls the models, clones `dunereco`, extracts
chunks, and embeds+indexes them into the shared Qdrant, then exits. Every step is
idempotent — re-running `provision` is a fast no-op unless the corpus changed, so
it doubles as a safe "refresh". Re-indexing is **safe while people are querying**:
it builds a fresh collection and atomically swaps an alias, so live queries never
see a half-built index (details in `EGEpipeline/qdrant_index.py`).

**GPU: required on every `up`, and it fails silently.**

On QuarkyLab's RTX 8000, generation is ~21s per query with the GPU vs ~74s on
CPU (~3.5x). **The override is not sticky** — any later `up -d` without
`-f docker-compose.gpu.yml` recreates Ollama in CPU-only mode with **no error
message**. It just gets slower. This has already happened once on this
deployment and went unnoticed until someone checked, so verify rather than assume.

Verify after any `up`, because the failure is silent:
```
curl -s localhost:11434/api/ps | python3 -m json.tool   # size_vram should equal size
nvidia-smi --query-gpu=memory.used --format=csv         # ~27 GB with both models loaded
docker logs arid-shared-ollama-1 2>&1 | grep 'inference compute'
```
That last line should say `library=CUDA ... Quadro RTX 8000`. If it says
`library=cpu` with `total_vram="0 B"`, you're on CPU.

Requires the NVIDIA driver plus
[`nvidia-container-toolkit`](https://github.com/NVIDIA/nvidia-container-toolkit)
registered with Docker (`docker info` should list an `nvidia` runtime).

<details>
<summary><b>Rootless Docker: registering the GPU runtime (no root needed)</b></summary>

QuarkyLab runs Docker rootless (`dockerd-rootless.sh` under a user account), and
a rootless daemon does **not** pick up the system `/etc/docker/daemon.json` — so
even with the toolkit installed system-wide, `docker info` shows no `nvidia`
runtime and every container silently lands on CPU. Both fixes are per-user, no
`sudo` required:

```
# 1. rootless can't manage cgroups -- the toolkit must be told to skip them
mkdir -p ~/.config/nvidia-container-runtime
printf '[nvidia-container-cli]\nno-cgroups = true\n' \
  > ~/.config/nvidia-container-runtime/config.toml

# 2. register the runtime with the *user* daemon, not /etc/docker
nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json
systemctl --user restart docker

# 3. confirm
docker info --format '{{range $k,$v := .Runtimes}}{{$k}} {{end}}'   # expect: ... nvidia ...
```
These live in `$HOME`, outside this repo, so they don't survive a host rebuild or
a move to a different account — re-run them if `inference compute` reports `cpu`.
</details>

Without a GPU, leave the override out; the base services never request a GPU
device, so they can't fail to find one.

### 2. Query from your own clone (each researcher)

See the [Quickstart](#quickstart-for-researchers-on-a-shared-host) for the fast
path. In short: `./setup.sh` once (or just run the tool — `chat.py`/`answer.py`
self-bootstrap the venv on first run), then the scripts default to the shared
services on `localhost:6333` / `localhost:11434`, so there's nothing to configure
when they're on this host:
```
python3 EGEpipeline/answer.py "how is neutrino energy reconstructed?"
python3 EGEpipeline/chat.py
```
Retrieval is hybrid: the **dense** half reads the shared Qdrant over HTTP (chunk
text is stored in the payload, so nothing local is needed for it); the **BM25**
half reads a local `chunks.jsonl`, which the **first query builds automatically**
by pulling chunk text back out of the shared Qdrant payloads — no `dunereco` clone,
tree-sitter grammars, or C compiler needed, and byte-identical to what `extract.py`
would have written. Without it, retrieval runs dense-only (still fully functional).
Run `python3 EGEpipeline/sync_chunks.py` by hand only if you'd rather pre-build it
than pay that one-time cost on the first question, or to refresh it after the
shared index is re-provisioned.

Generation runs against the shared Ollama — no login or API key needed. If the
shared Ollama is unreachable or the model isn't pulled, queries return the
retrieved **sources** with a clear message instead of crashing.

**Interactive multi-turn CLI for researchers:**
```
python3 EGEpipeline/chat.py
```

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

Useful env vars (export them in your shell; there's no config file to edit). The
only ones most people ever touch are `QDRANT_URL` / `OLLAMA_HOST`, and only if the
shared stack lives on **another machine** — otherwise the `localhost` defaults just
work:
| Var | Default | Purpose |
|-----|---------|---------|
| `QDRANT_URL` | `http://localhost:6333` | shared Qdrant endpoint the scripts query |
| `OLLAMA_HOST` | `http://localhost:11434` | shared Ollama endpoint for embeds/generation |
| `QDRANT_COLLECTION` | `dunereco` | read/alias name in Qdrant (change only to serve a different corpus) |
| `DUNERECO_URL` | `github.com/DUNE/dunereco.git` | corpus to clone (provision) |
| `DUNERECO_REF` | repo default branch | branch/tag to check out (provision) |
| `FORCE_EXTRACT` | unset | `1` re-extracts `chunks.jsonl` even if present (provision) |
| `FORCE_EMBED` | unset | `1` forces a rebuild even if the shared index already matches (provision) |
| `EMBED_WORKERS` | `8` | concurrent batched embed requests. Saturates at ~4 on one GPU — raising it does nothing (see [Performance](#performance-and-known-limits)) |
| `ARID_ORPHAN_KEEP` | `3600` (1 hr) | seconds an unreferenced old collection survives before cleanup |

Match `EMBED_WORKERS` to `OLLAMA_NUM_PARALLEL` on the shared Ollama service
(`docker-compose.shared.yml`, currently `8`) — extra client threads just queue
server-side.

### Shared vs. isolated: why this layout

The services are shared on purpose. On a single-GPU box (e.g. one 46 GB card),
each Ollama instance that serves the 30B generation model holds ~19 GB of VRAM,
and separate containers don't share loaded weights — so N researchers each
running their own Ollama would need N × ~19 GB and fill the card after two or
three. Running **one** shared Ollama loads the model once and serves everyone
concurrently (`OLLAMA_NUM_PARALLEL`), and **one** shared Qdrant means the corpus
is embedded once, not per person. Chunk text lives in the Qdrant payload, so
querying researchers need nothing on disk except (optionally) `chunks.jsonl` for
the BM25 half — no shared filesystem required.

If you genuinely need a fully isolated stack (a different corpus, or an offline
box), you can still run everything self-contained: bring up
`docker-compose.shared.yml` under your own project name and provision it against
its own volumes.

### Performance and known limits

Measured on QuarkyLab (Quadro RTX 8000, 46 GB; rootless Docker) against the 7004-chunk
dunereco corpus.

| | GPU | CPU |
|---|---|---|
| generation, one query | ~21s | ~74s |
| full re-embed, 7004 chunks | ~3m45s | ~40min |

**Embedding is GPU-token-bound, and already at its ceiling.** Cost tracks the
total number of characters sent, not how they're packaged. Things that were
measured and do **not** help:
- **`BATCH_SIZE`** (currently 32): flat from 32 to 512. Collapsing 64 requests
  into 4 changed throughput by <1%, so per-request overhead isn't the bottleneck.
- **Sorting chunks by length** to reduce padding waste: Ollama doesn't pad
  batches to their longest member, so there's nothing to reclaim.
- **More `EMBED_WORKERS`**: saturates at ~4; 8 is already past the knee.

The only knob that moves the needle is `MAX_EMBED_CHARS` in `embed_store.py`
(currently 6000), because it directly reduces tokens — the corpus is heavily
skewed, with the longest 5% of chunks accounting for ~47% of all embedded
characters. Lowering it to 2000 cuts embed time ~23%, but truncates the tail of
your largest functions out of the *embedding* (full text is still stored and
still searched by BM25). Not recommended: it trades retrieval quality on big
functions for a couple of minutes on an infrequent job.

**Known limitations, not yet addressed:**
- **Re-embedding is all-or-nothing.** A corpus fingerprint makes re-provisioning a
  no-op when *nothing* changed, but a single changed chunk re-embeds all 7004.
  Since the previous collection still holds valid vectors for unchanged chunks,
  reusing them and embedding only the diff would make typical updates near-instant
  with no quality cost. This is the biggest available win.
- **`TOP_K = 6` can truncate the good result.** For a query like
  "what does NeutrinoEnergyRecoAlg do?", six near-duplicate one-line `.fcl` config
  entries crowd out the actual `NeutrinoEnergyRecoAlg.cc` implementation, which
  hybrid retrieval ranks 7th — so generation sees only config and correctly reports
  it can't describe the algorithm. A `TOP_K` bump or a per-file diversity cap would
  fix it, at some context-window cost.

### Manual / non-Docker setup

Still works if you'd rather run things directly:
```
pip install -r requirements.txt          # needs a C compiler (tree-sitter grammars)
ollama pull qwen3-embedding:0.6b         # requires Ollama running locally
ollama pull qwen3-coder:30b              # generation model used by answer.py / arid_mcp.py
docker run -p 6333:6333 qdrant/qdrant    # requires Qdrant running locally
git clone https://github.com/DUNE/dunereco.git
python extract.py dunereco/                        # writes chunks.jsonl
python EGEpipeline/embed_store.py chunks.jsonl      # safe rebuild: builds a fresh collection + swaps the alias
python EGEpipeline/answer.py "your question"
```
`embed_store.py` takes `--force` (rebuild even if the store already matches these
chunks). BM25-only mode needs neither Ollama nor Qdrant — just `pip install`
and skip straight to `search_bm25.py`.

---

## File Structure

```
ARID/
├── setup.sh                One-time researcher setup: builds the query venv (idempotent)
├── extract.py              Chunk extraction entry point
├── chunk_schema.py         Schema definition + validation
├── chunks.jsonl            Output of extract.py (gitignored if large)
├── requirements.txt        Full deps (incl. tree-sitter grammars; needs a C compiler)
├── requirements-query.txt  Query-only deps for researchers (qdrant-client + ollama)
├── tree-sitter-jsonnet/    Vendored jsonnet grammar (patched setup.py; built by requirements.txt)
├── parsers/
│   ├── parse_cpp.py        C++ parser (tree-sitter)
│   ├── parse_python.py     Python parser (tree-sitter)
│   ├── parse_jsonnet.py    Jsonnet parser (tree-sitter)
│   └── parse_fcl.py        FHiCL parser (in-house recursive-descent lexer + parser)
├── EGEpipeline/
│   ├── env_setup.py        ensure_env(): entry points self-bootstrap the venv + re-exec into it
│   ├── embed_store.py      Embedding + dense search + safe (re)build entry point
│   ├── qdrant_index.py     Concurrency-safe indexing: alias + fresh-build + atomic swap + lock
│   ├── sync_chunks.py      Rebuild local chunks.jsonl from Qdrant payloads (no compiler needed)
│   ├── search_bm25.py      BM25 keyword search
│   ├── search.py           Hybrid RRF fusion (dense + BM25); degrades if either half is down
│   ├── answer.py           RAG answer step (retrieve → generate → cite)
│   ├── chat.py             Interactive multi-turn CLI with researcher rating + logging
│   └── arid_mcp.py         MCP server exposing search_arid/ask_arid over SSH stdio
├── feedback/                gitignored: chat.py session logs, one JSON per researcher/session
├── tests/                  pytest suite for the parsers + chunk schema
├── eval/
│   └── gold_queries.json   Gold query set for retrieval eval (KAN-24)
├── docker/
│   └── bootstrap.py        Provisioner: clone → extract → safe embed+index, idempotent
├── services.sh             Detect-or-host the shared services (reuse if running, else up + provision)
├── docker-compose.shared.yml  the ONE shared stack: qdrant + ollama + provision (run once per host)
├── docker-compose.gpu.yml  optional override: adds NVIDIA GPU passthrough to ollama
├── arid-mcp.cmd            Windows shim: sshd forced command for the MCP stdio channel
├── Dockerfile
└── dunereco/               Clone of the target repo (gitignored, not baked into the image)
```

---

## Evaluation

`eval/gold_queries.json` contains a starter set of 7 natural-language queries with expected chunk targets for measuring retrieval quality (Recall@k, MRR). All queries are currently `unverified-name-match` status — targets were derived by pattern-matching symbol names and need confirmation against actual function bodies (KAN-24).

---

# Changelog

Notable changes to ARID, newest first. Every change to this repo should get an
entry here — see `CLAUDE.md`.

## 2026-07-09

- **One-command containerized install.** `docker compose up -d --build` now
  bootstraps the whole pipeline (clone `dunereco` → extract → embed+index)
  idempotently via `docker/bootstrap.py`. Safe to re-run any time — already-done
  steps are skipped (checked against disk / the Qdrant collection).
- **GPU passthrough made opt-in.** Moved the NVIDIA device reservation out of
  the base `docker-compose.yml` into `docker-compose.gpu.yml`, so the base file
  works on any machine with Docker, not just ones with an NVIDIA GPU +
  `nvidia-container-toolkit`. Add `-f docker-compose.gpu.yml` to use it.
- **Batched + concurrent embedding.** `EGEpipeline/embed_store.py` switched
  from one `ollama.embeddings()` call per chunk to the batched `ollama.embed()`
  endpoint, run `EMBED_WORKERS` (default 8) batches at a time via a thread
  pool. `OLLAMA_NUM_PARALLEL=8` set on the Ollama service so it actually
  processes those requests concurrently instead of queueing them. Full
  dunereco corpus (7004 chunks) indexing time: ~40 min (CPU, sequential) →
  9m30s (GPU, sequential) → **3m39s** (GPU, batched + concurrent).

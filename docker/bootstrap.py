"""Container entrypoint: clone dunereco, extract chunks, embed+index -- then idle.

Runs on every `docker compose up`, but every step is idempotent (checks what's
already on disk / in Qdrant before doing anything), so a re-run after the first
successful one is just a fast no-op. That's what makes `docker compose up
--build` the whole install: fresh clone of ARID -> one command -> ready to query.

Env overrides:
    DUNERECO_URL    git URL to clone           (default: DUNE/dunereco on GitHub)
    DUNERECO_REF    branch/tag to check out    (default: repo's default branch)
    FORCE_EXTRACT   "1" re-extracts chunks.jsonl even if it already exists
    FORCE_EMBED     "1" re-embeds even if the Qdrant collection already has points
"""

import os
import subprocess
import sys
import time

APP_DIR = "/app"
DUNERECO_URL = os.environ.get("DUNERECO_URL", "https://github.com/DUNE/dunereco.git")
DUNERECO_REF = os.environ.get("DUNERECO_REF", "")
FORCE_EXTRACT = os.environ.get("FORCE_EXTRACT") == "1"
FORCE_EMBED = os.environ.get("FORCE_EMBED") == "1"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def log(msg: str):
    print(f"[bootstrap] {msg}", flush=True)


def run(*args: str):
    log("$ " + " ".join(args))
    subprocess.run(args, check=True)


def pull_models():
    import ollama
    for model in ("qwen3-embedding:0.6b", "qwen3-coder:30b"):
        log(f"pulling {model} (idempotent, skips if present)...")
        ollama.pull(model)


def clone_dunereco():
    if os.path.isdir(os.path.join(APP_DIR, "dunereco", ".git")):
        log("dunereco/ already present, skipping clone "
            "(rm -rf dunereco/ or bump DUNERECO_REF to refresh)")
        return
    log("cloning dunereco...")
    args = ["git", "clone", "--depth", "1"]
    if DUNERECO_REF:
        args += ["--branch", DUNERECO_REF]
    args += [DUNERECO_URL, "dunereco"]
    run(*args)


def extract_chunks():
    chunks_path = os.path.join(APP_DIR, "chunks.jsonl")
    already_extracted = os.path.exists(chunks_path) and os.path.getsize(chunks_path) > 0
    if not FORCE_EXTRACT and already_extracted:
        log("chunks.jsonl already present, skipping extract (set FORCE_EXTRACT=1 to redo)")
        return
    log("extracting chunks from dunereco/...")
    run(sys.executable, "extract.py", "dunereco/")


def wait_for_qdrant(retries: int = 30, delay: float = 1.0):
    from qdrant_client import QdrantClient
    client = QdrantClient(url=QDRANT_URL)
    for attempt in range(retries):
        try:
            client.get_collections()
            return client
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


def embed_and_index():
    # Wait for Qdrant, then hand off to embed_store.py's SAFE rebuild: it
    # fingerprints the corpus and skips if the shared index already matches
    # (so re-running provision is a fast no-op), otherwise builds into a fresh
    # collection and atomically swaps the alias -- concurrent queries never see a
    # partial index, and an interrupted run leaves the live index untouched.
    # See EGEpipeline/qdrant_index.py. FORCE_EMBED=1 forces a rebuild anyway.
    wait_for_qdrant()
    log("embedding + indexing chunks (safe rebuild; skips if already current)...")
    args = [sys.executable, "EGEpipeline/embed_store.py", "chunks.jsonl"]
    if FORCE_EMBED:
        args.append("--force")
    run(*args)


def main():
    os.chdir(APP_DIR)
    pull_models()
    clone_dunereco()
    extract_chunks()
    embed_and_index()

    log("provisioning complete. Query the shared services with:")
    log('  python EGEpipeline/search.py "your query"')
    log('  python EGEpipeline/answer.py "your query"')
    log('  python EGEpipeline/chat.py')


if __name__ == "__main__":
    main()

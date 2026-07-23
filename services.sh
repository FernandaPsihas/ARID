#!/usr/bin/env bash
# Ensure the shared ARID services (Qdrant + Ollama) exist, then make sure the
# dunereco corpus is indexed. Detect-or-host, idempotent -- safe to re-run:
#
#   ./services.sh
#
#   * Qdrant (6333) and Ollama (11434) already reachable -> reuse them as the
#     shared instance; start nothing.
#   * Either one missing -> stand up the shared stack (docker-compose.shared.yml),
#     automatically adding the NVIDIA GPU override when this host has a usable GPU
#     (so `up` is never accidentally CPU-only).
#   * Corpus not indexed yet -> provision it (clone + extract + embed). Already
#     indexed -> skip.
#
# Researchers who just want to query a host where the services already run don't
# need this at all -- this is the one-time (re)start for whoever OWNS the services.
# To query:  python3 EGEpipeline/chat.py
set -euo pipefail
cd "$(dirname "$0")"

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
COLLECTION="${QDRANT_COLLECTION:-dunereco}"

_probe()    { curl -sf -o /dev/null --max-time 3 "$1"; }
qdrant_up() { _probe "$QDRANT_URL/readyz"; }
ollama_up() { _probe "$OLLAMA_HOST/api/version"; }

# GPU override, automatically -- a GPU host must never silently run Ollama on CPU.
compose=(-f docker-compose.shared.yml)
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 \
   && docker info 2>/dev/null | grep -qi nvidia; then
  compose+=(-f docker-compose.gpu.yml)
  echo "==> NVIDIA GPU + docker runtime detected -> GPU mode."
else
  echo "==> No usable GPU/runtime -> CPU mode."
fi

# --- services: reuse if already running, else host them ----------------------
if qdrant_up && ollama_up; then
  echo "==> Qdrant + Ollama already running -> reusing them as the shared instance."
else
  echo "==> Services not fully up (qdrant=$(qdrant_up && echo up || echo down)," \
       "ollama=$(ollama_up && echo up || echo down)) -> hosting them."
  docker compose "${compose[@]}" up -d qdrant ollama
  printf "    waiting for readiness"
  for _ in $(seq 1 60); do
    if qdrant_up && ollama_up; then printf " ready\n"; break; fi
    printf "."; sleep 2
  done
  qdrant_up && ollama_up || { printf "\nERROR: services did not become ready in time\n" >&2; exit 1; }
fi

# --- corpus: provision only if the collection/alias isn't there yet ----------
if _probe "$QDRANT_URL/collections/$COLLECTION"; then
  echo "==> Corpus '$COLLECTION' already indexed -> nothing to provision."
else
  echo "==> Corpus '$COLLECTION' not indexed -> provisioning (clone + extract + embed)..."
  docker compose "${compose[@]}" run --rm provision
fi

echo
echo "Shared services ready. Query with:  python3 EGEpipeline/chat.py"

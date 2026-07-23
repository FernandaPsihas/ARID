#!/usr/bin/env bash
# ARID one-time setup: build the query virtualenv. Safe to re-run (idempotent).
#
#   git clone <this repo> ARID && cd ARID && ./setup.sh
#
# After this, just ask -- no activation, no separate chunk sync:
#   python3 EGEpipeline/chat.py
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
VENV=".venv"

if [ ! -x "$VENV/bin/python" ]; then
  echo "==> creating venv ($VENV)"
  # Plain `python3 -m venv` fails on hosts without the python3-venv/ensurepip
  # package (e.g. QuarkyLab): it can't bootstrap pip. Fall back to building the
  # venv without pip and fetching it manually. This branch works everywhere.
  if ! "$PY" -m venv "$VENV" >/dev/null 2>&1; then
    echo "    (no ensurepip on this host; bootstrapping pip manually)"
    rm -rf "$VENV"
    "$PY" -m venv --without-pip "$VENV"
    curl -sS https://bootstrap.pypa.io/get-pip.py | "$VENV/bin/python"
  fi
fi

echo "==> installing query dependencies (requirements-query.txt)"
"$VENV/bin/pip" install -q -r requirements-query.txt

# Readiness stamp: env_setup.ensure_env() treats the venv as usable only if this
# exists, so an interrupted setup (no stamp) gets rebuilt instead of half-used.
# set -e above guarantees we only reach this line if everything above succeeded.
touch "$VENV/.arid-setup-complete"

echo
echo "Done. Ask a question with:"
echo "    python3 EGEpipeline/chat.py                              # interactive"
echo "    python3 EGEpipeline/answer.py \"how are pixel maps built?\"  # one-shot"

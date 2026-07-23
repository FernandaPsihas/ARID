"""Shared environment guard for the ARID query tools.

Any entry-point script calls `ensure_env()` as its very first action -- before
importing anything third-party -- and is then guaranteed to be running under the
repo's `.venv` with the query dependencies installed. Three cases:

  * no `.venv` yet          -> runs `setup.sh` to build it, then re-execs into it
  * `.venv` exists, not in it -> re-execs into it (so `source activate` is optional)
  * already in the `.venv`    -> returns immediately, nothing to do

Kept stdlib-only on purpose: it has to run under whatever interpreter launched
the script -- usually the system `python3`, before the venv (and its packages)
even exist -- so it can't rely on anything pip-installed.

Readiness is gated on a stamp file (`.venv/.arid-setup-complete`) that `setup.sh`
writes only after a successful install, not just on the venv directory existing.
That way a half-built venv from an interrupted setup is treated as "not ready"
and rebuilt, instead of being re-exec'd into and blowing up on a missing import.
"""

import os
import subprocess
import sys
from pathlib import Path

_WIN = os.name == "nt"
_REPO = Path(__file__).resolve().parent.parent
_VENV = _REPO / ".venv"
_VENV_PY = _VENV / ("Scripts/python.exe" if _WIN else "bin/python")
_STAMP = _VENV / ".arid-setup-complete"   # written by setup.sh on success
_GUARD = "ARID_ENV_BOOTSTRAPPED"          # env flag that breaks any re-bootstrap loop


def _in_venv() -> bool:
    # Compare prefixes, not the python binaries: a venv's python is a symlink to
    # the base interpreter, so resolving the executables makes them look identical.
    return Path(sys.prefix).resolve() == _VENV.resolve()


def _env_ready() -> bool:
    return _VENV_PY.exists() and _STAMP.exists()


def _bootstrap() -> None:
    setup = _REPO / "setup.sh"
    if _WIN or not setup.exists():
        sys.exit(
            f"ARID: environment isn't set up and I can't self-bootstrap here "
            f"(missing {setup.name} or unsupported platform).\n"
            f"Set it up manually and retry."
        )
    print("ARID: setting up the query environment (one-time, ~30s)...", file=sys.stderr)
    try:
        subprocess.run(["bash", str(setup)], check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"ARID: setup.sh failed (exit {e.returncode}); can't continue.")


def ensure_env() -> None:
    """Guarantee we're running under the repo venv with deps installed."""
    if _in_venv() and _env_ready():
        return                          # fast path: already good
    if not _env_ready():
        if os.environ.get(_GUARD):      # we already built it once this launch --
            sys.exit("ARID: environment still incomplete after setup; aborting.")
        _bootstrap()
    if _VENV_PY.exists() and not _in_venv():
        os.environ[_GUARD] = "1"        # mark so the re-exec'd copy won't loop
        os.execv(str(_VENV_PY), [str(_VENV_PY), *sys.argv])


if __name__ == "__main__":
    # Smoke test / manual check: set up if needed, then report where we landed.
    ensure_env()
    print(f"ok -- running under {sys.prefix}")

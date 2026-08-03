"""Clone/update the indexed repos and extract them into one corpus file.

extract.py emits repo-relative paths, which are only unique WITHIN a repo. Across
24 repos `CMakeLists.txt` and `Utilities/Foo.h` collide, and chunk ids feed uuid5
-> Qdrant point id, so a collision silently overwrites the earlier chunk rather
than erroring. Both `file` and `id` therefore get a repo prefix here.

Usage:
    python build_corpus.py                  # update clones, extract, write chunks.jsonl
    python build_corpus.py --no-pull        # extract from what's already on disk
    python build_corpus.py --repos-dir DIR  # default: ./repos (gitignored)

Then index it:
    python EGEpipeline/embed_store.py chunks.jsonl                 # full rebuild
    python EGEpipeline/embed_store.py chunks.jsonl --incremental   # reuse unchanged
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chunk_schema import validate_chunk

ARID = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(ARID, ".venv", "bin", "python")

# Tier 1 -- the core reconstruction chain: raw data -> sim -> reco -> data products
# -> analysis, plus the meta-repos whose build recipes double as a map of how it all
# links together. A plain list because adding a repo is a one-line edit and a config
# file would just be this list with extra steps; if this grows past a few tiers or
# needs per-repo settings, promote it then.
#
# larrecoalg is deliberately absent: upstream it's an empty stub, the content lives
# in larreco/larreco/RecoAlg/. See INDEX_EXCLUSIONS.md.
REPOS = [
    # DUNE
    "https://github.com/DUNE/dunecore",
    "https://github.com/DUNE/dunesim",
    "https://github.com/DUNE/dunereco",
    "https://github.com/DUNE/duneana",
    "https://github.com/DUNE/duneopdet",
    "https://github.com/DUNE/dunedataprep",
    "https://github.com/DUNE/dunecalib",
    "https://github.com/DUNE/duneanaobj",
    "https://github.com/DUNE/duneutil",
    "https://github.com/DUNE/dunesw",
    # LArSoft
    "https://github.com/LArSoft/larcore",
    "https://github.com/LArSoft/larcorealg",
    "https://github.com/LArSoft/larcoreobj",
    "https://github.com/LArSoft/lardata",
    "https://github.com/LArSoft/lardataalg",
    "https://github.com/LArSoft/lardataobj",
    "https://github.com/LArSoft/larevt",
    "https://github.com/LArSoft/larsim",
    "https://github.com/LArSoft/larg4",
    "https://github.com/LArSoft/larreco",
    "https://github.com/LArSoft/larana",
    "https://github.com/LArSoft/larpandora",
    "https://github.com/LArSoft/larpandoracontent",
    "https://github.com/LArSoft/larsoft",
]


def sync(url: str, dest: str, pull: bool) -> None:
    """Shallow-clone `url` to `dest`, or fast-forward it if it's already there."""
    name = os.path.basename(dest)
    if not os.path.isdir(os.path.join(dest, ".git")):
        print(f"{name:<22} cloning", flush=True)
        subprocess.run(["git", "clone", "--depth", "1", url, dest],
                       check=True, capture_output=True,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    elif pull:
        # --depth 1 keeps these shallow; a plain pull would drag in full history.
        subprocess.run(["git", "fetch", "--depth", "1", "origin"],
                       cwd=dest, check=True, capture_output=True)
        subprocess.run(["git", "reset", "--hard", "origin/HEAD"],
                       cwd=dest, check=True, capture_output=True)


def extract_repo(repo_dir: str, scratch: str) -> list[dict]:
    """Run extract.py over one repo and return its chunks, repo-prefixed."""
    repo = os.path.basename(repo_dir)
    subprocess.run([PYTHON, os.path.join(ARID, "extract.py"), repo_dir],
                   cwd=scratch, check=True, capture_output=True)
    out = []
    with open(os.path.join(scratch, "chunks.jsonl"), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            # Prefix the existing id rather than recomputing it with make_chunk_id:
            # extract.py may have appended a "#n" collision disambiguator, and
            # recomputing would drop it. The id starts with the repo-relative file
            # path, so prefixing is equivalent.
            assert c["id"].startswith(c["file"]), (c["id"], c["file"])
            c["id"] = f"{repo}/{c['id']}"
            c["file"] = f"{repo}/{c['file']}"
            errs = validate_chunk(c)
            assert not errs, (c["id"], errs)
            out.append(c)
    return out


def main() -> None:
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    pull = "--no-pull" not in flags
    repos_dir = os.environ.get("ARID_REPOS_DIR", os.path.join(ARID, "repos"))
    if "--repos-dir" in sys.argv:
        repos_dir = sys.argv[sys.argv.index("--repos-dir") + 1]
    os.makedirs(repos_dir, exist_ok=True)

    scratch = os.path.join(repos_dir, ".extract")
    os.makedirs(scratch, exist_ok=True)
    out_path = os.path.join(ARID, "chunks.jsonl")

    seen: set[str] = set()
    total = 0
    # Write to a temp file and rename at the end: a crash partway through must not
    # leave a truncated chunks.jsonl behind, since BM25 reads this file directly
    # (see search.py CHUNKS_PATH) and would silently serve a partial corpus.
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as out:
        for url in REPOS:
            name = url.rstrip("/").split("/")[-1]
            dest = os.path.join(repos_dir, name)
            sync(url, dest, pull)
            chunks = extract_repo(dest, scratch)
            for c in chunks:
                assert c["id"] not in seen, f"duplicate chunk id: {c['id']}"
                seen.add(c["id"])
                out.write(json.dumps(c) + "\n")
            total += len(chunks)
            print(f"{name:<22} {len(chunks):>7}", flush=True)
    os.replace(tmp_path, out_path)

    print(f"\n{len(REPOS)} repos, {total} chunks -> {out_path}")


if __name__ == "__main__":
    main()

"""Test extract.py (KAN-10..13) — discovery, dispatch, relative paths, fallback."""

import json
import os
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")  # ponytail: Windows console is cp1252, ✓ needs utf-8
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import extract

FILES = {
    "code/foo.py": "def greet(name):\n    return name\n",
    "code/bar.cc": "int add(int a, int b) {\n    return a + b;\n}\n",
    "cfg/job.fcl": "physics: {\n    producer: x\n}\n",
    # nothing parseable (comment only) -> parser returns [] -> whole-file fallback
    "cfg/flat.fcl": "# just a comment, nothing to parse here\n",
}


def _git(root, *args):
    subprocess.run(["git", "-C", root, *args], check=True, capture_output=True)


def test_extract():
    with tempfile.TemporaryDirectory() as root:
        for rel, body in FILES.items():
            p = os.path.join(root, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(body)
        # extract relies on git ls-files, so the fixture must be a repo with commits
        _git(root, "init", "-q")
        _git(root, "add", "-A")
        _git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x")

        cwd = os.getcwd()
        os.chdir(root)
        try:
            result = extract.extract(root, [])
            chunks = [json.loads(l) for l in open(extract.OUT, encoding="utf-8")]
        finally:
            os.chdir(cwd)

    stats = result["stats"]
    assert stats["python"][0] == 1, stats
    assert stats["cpp"][0] == 1, stats
    assert stats["fcl"][0] == 2, stats

    # every file/id path is repo-relative (forward slashes, no temp prefix)
    for c in chunks:
        assert not os.path.isabs(c["file"]), c["file"]
        assert c["id"].startswith(c["file"]), c

    # flat.fcl had no braces -> exactly one whole-file fallback chunk
    flat = [c for c in chunks if c["file"].endswith("flat.fcl")]
    assert len(flat) == 1 and flat[0]["symbol"] == extract.FALLBACK_SYMBOL, flat
    assert "cfg/flat.fcl" in result["fallback_files"]

    # job.fcl had a real brace block -> NOT a fallback
    job = [c for c in chunks if c["file"].endswith("job.fcl")]
    assert job and job[0]["symbol"] != extract.FALLBACK_SYMBOL, job

    print("KAN-10..13 ✓")


EXCL_FILES = {
    # lowercase .h must dispatch to the cpp parser (Tier 1 is 1636 .h / 0 .H)
    "inc/Hit.h": "class Hit {\npublic:\n    int View() const { return v; }\n};\n",
    "src/keep.cc": "int keep() {\n    return 1;\n}\n",
    "cfg/gen_real.fcl": "physics: {\n    producer: keepme\n}\n",
    # one file per EXCLUDE_PATTERNS rule, all otherwise perfectly parseable
    "test/unit.cc": "int dropped() {\n    return 0;\n}\n",
    "larreco/Genfit/Fit.cc": "int vendored() {\n    return 0;\n}\n",
    "fcl/protodune/fcldirs/calib/g.fcl": "Values: [\n    0.19, 0.19\n]\n",
    "cfg/prodgenie_dune10kt.fcl": "physics: {\n    producer: perm\n}\n",
}


def test_exclusions_and_header_dispatch():
    with tempfile.TemporaryDirectory() as root:
        for rel, body in EXCL_FILES.items():
            p = os.path.join(root, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(body)
        _git(root, "init", "-q")
        _git(root, "add", "-A")
        _git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x")

        cwd = os.getcwd()
        os.chdir(root)
        try:
            result = extract.extract(root, [])
            chunks = [json.loads(l) for l in open(extract.OUT, encoding="utf-8")]
        finally:
            os.chdir(cwd)

    indexed = {c["file"] for c in chunks}

    # the regression this guards: .h reaching parse_cpp at all
    assert "inc/Hit.h" in indexed, sorted(indexed)
    assert result["stats"]["cpp"][0] == 2, result["stats"]  # Hit.h + keep.cc

    # every rule fired exactly once, and nothing it matched got indexed
    assert result["excluded"] == {
        "test-dirs": 1,
        "vendored-genfit": 1,
        "calib-constant-tables": 1,
        "prod-job-permutations": 1,
    }, result["excluded"]
    for rel in ("test/unit.cc", "larreco/Genfit/Fit.cc",
                "fcl/protodune/fcldirs/calib/g.fcl", "cfg/prodgenie_dune10kt.fcl"):
        assert rel not in indexed, rel

    # non-prod fcl next to an excluded one survives (rule is prefix-scoped)
    assert "cfg/gen_real.fcl" in indexed, sorted(indexed)

    print("exclusions + .h dispatch ✓")


# one-line class carrying its own inline ctor -> parse_cpp emits two chunks with
# the same symbol at the same line, so make_chunk_id returns the same id for both
COLLIDE = {"inc/Hw.h": "class Cryostat : public Element{ public: Cryostat(ID id) : Element(id) {} };\n"}


def test_chunk_ids_are_unique():
    with tempfile.TemporaryDirectory() as root:
        for rel, body in COLLIDE.items():
            p = os.path.join(root, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(body)
        _git(root, "init", "-q")
        _git(root, "add", "-A")
        _git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x")

        cwd = os.getcwd()
        os.chdir(root)
        try:
            result = extract.extract(root, [])
            chunks = [json.loads(l) for l in open(extract.OUT, encoding="utf-8")]
        finally:
            os.chdir(cwd)

    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), sorted(ids)
    if result["collisions"]:
        assert any("#" in i for i in ids), sorted(ids)

    print("unique chunk ids ✓")


if __name__ == "__main__":
    test_extract()
    test_exclusions_and_header_dispatch()
    test_chunk_ids_are_unique()

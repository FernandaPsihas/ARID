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
    # no braces anywhere -> parser returns [] -> must become a whole-file fallback
    "cfg/flat.fcl": "#include \"base.fcl\"\nthreshold: 0.5\n",
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


if __name__ == "__main__":
    test_extract()

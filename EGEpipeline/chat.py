"""Interactive multi-turn CLI for researchers to query dunereco and rate answers.

Flow: ask for a name, then loop taking questions until the researcher quits
(Ctrl+D, or typing quit/exit). Each answer streams live via answer.py's
_generate(), keeping conversational memory across turns (prior Q&A text, plus
the code chunks from the last turn -- see CONTEXT_CAP below. A follow-up like
"what about the flip handling in that function?" doesn't restate the symbol
it means, so a from-scratch retrieval on the follow-up alone can miss the
chunk that's still sitting right there from the previous turn; carrying it
forward fixes that without needing history-aware query rewriting). After
each answer the researcher rates it good / needs improvement (with reason
codes, or their own words) and the whole session -- every question, answer,
sources, and rating -- is saved to feedback/<researcher>/<timestamp>.json
after every turn, so a crash or Ctrl+C never loses more than the in-flight
turn.

Usage:
    python EGEpipeline/chat.py
"""

import json
import os
import sys
import time
from datetime import datetime

# Guarantee we're running under the repo .venv with deps installed (building it
# on first run if needed) BEFORE importing anything third-party. Shared across
# the entry points via env_setup; may re-exec the process, so keep it first.
sys.path.insert(0, os.path.dirname(__file__))
from env_setup import ensure_env
ensure_env()

from answer import GEN_MODEL, TOP_K, GenerationUnavailable, _generate, _retrieve

FEEDBACK_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "feedback")
CHAT_NUM_CTX = 16384    # bumped over answer.py's 8192: multi-turn history adds a bit of
                        # length, and there's VRAM headroom to spare (see CHANGELOG.md)
MAX_HISTORY_TURNS = 8   # plain-text Q&A pairs kept in the model's context; the saved
                        # transcript keeps every turn regardless of this cap
CONTEXT_CAP = TOP_K * 2 # carried-forward + fresh chunks, capped so a long session
                        # can't pile snippets up unboundedly (7/20 meeting item 5)

REASONS = [
    ("missing_incomplete_info", "Missing or incomplete information"),
    ("wrong_citations", "Wrong or irrelevant citations"),
    ("inaccurate_hallucinated", "Inaccurate or hallucinated"),
    ("too_vague", "Too vague / not specific enough"),
    ("retrieval_missed_code", "Retrieval missed the relevant code"),
    ("other", "Other (describe below)"),
]

QUIT_WORDS = {"quit", "exit", "/quit", "/exit", "/q"}


def _merge_chunks(previous: list[dict], fresh: list[dict], cap: int) -> list[dict]:
    """Previous turn's chunks first (still relevant, and what a vague follow-up
    like "what about X in that function?" actually needs), then fresh results
    not already present, truncated to cap so a long session can't grow the
    context forever."""
    merged = list(previous)
    seen = {c["id"] for c in previous}
    for c in fresh:
        if c["id"] not in seen:
            merged.append(c)
            seen.add(c["id"])
    return merged[:cap]


def slugify(name: str) -> str:
    import re
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return s or "researcher"


def session_path(researcher_slug: str, started_at: datetime, base_dir: str = FEEDBACK_DIR) -> str:
    d = os.path.join(base_dir, researcher_slug)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, started_at.strftime("%Y%m%d_%H%M%S") + ".json")


def save_session(session: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(session, f, indent=2)
    os.replace(tmp, path)  # atomic -- a crash mid-write can't corrupt the saved file


def ask_name() -> str:
    while True:
        name = input("Your name: ").strip()
        if name:
            return name
        print("Please enter a name.")


def prompt_rating() -> tuple[str, list[str], str]:
    while True:
        choice = input("\nRate this answer  [g]ood / [n]eeds improvement / [s]kip: ").strip().lower()
        if choice in ("g", "good"):
            return "good", [], ""
        if choice in ("n", "needs improvement", "needs_improvement"):
            break
        if choice in ("s", "skip", ""):
            return "skipped", [], ""
        print("Please enter g, n, or s.")

    print("What needs improvement? (comma-separated numbers)")
    for i, (_, label) in enumerate(REASONS, 1):
        print(f"  [{i}] {label}")
    reasons: list[str] = []
    while not reasons:
        raw = input("> ").strip()
        idxs = set()
        for tok in raw.split(","):
            tok = tok.strip()
            if tok.isdigit() and 1 <= int(tok) <= len(REASONS):
                idxs.add(int(tok))
        if idxs:
            reasons = [REASONS[i - 1][0] for i in sorted(idxs)]
        else:
            print(f"Enter one or more numbers 1-{len(REASONS)}, comma-separated.")

    notes = input("Anything else to add? (optional, Enter to skip): ").strip()
    if reasons == ["other"] and not notes:
        notes = input("You picked 'Other' -- what would you suggest instead? ").strip()
    return "needs_improvement", reasons, notes


def main() -> None:
    print("=== ARID Research Assistant ===")
    print(f"Model: {GEN_MODEL}  |  type 'quit' or Ctrl+D to end the session\n")
    researcher = ask_name()
    started_at = datetime.now()
    path = session_path(slugify(researcher), started_at)
    session = {
        "researcher": researcher,
        "model": GEN_MODEL,
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": None,
        "turns": [],
    }
    print(f"\nHi {researcher}! Ask anything about the dunereco codebase.")
    print(f"Saving this session to {path}\n")

    history: list[dict] = []  # plain user/assistant text, no chunks -- see module docstring
    last_chunks: list[dict] = []  # carried forward each turn, see _merge_chunks
    turn_num = 0
    try:
        while True:
            try:
                question = input(f"[{turn_num + 1}] > ").strip()
            except EOFError:
                print()
                break
            if not question:
                continue
            if question.lower() in QUIT_WORDS:
                break

            t0 = time.time()
            fresh_chunks = _retrieve(question, top_k=TOP_K)
            chunks = _merge_chunks(last_chunks, fresh_chunks, CONTEXT_CAP)
            if not chunks:
                print("No relevant chunks found in the dunereco codebase.\n")
                continue

            sources = [{"file": c["file"], "start_line": c["start_line"],
                        "end_line": c["end_line"], "symbol": c["symbol"]} for c in chunks]
            src_text = "\n".join(
                f"  {c['file']}  L{c['start_line']}-{c['end_line']}  {c['symbol']}" for c in chunks)

            print()
            try:
                body = _generate(question, chunks, stream=True,
                                 history=history[-2 * MAX_HISTORY_TURNS:], num_ctx=CHAT_NUM_CTX)
            except GenerationUnavailable as e:
                # Keep the session alive: show what we retrieved and let them ask again.
                print(f"\n[generation unavailable] {e}")
                print(f"\nSources (retrieval still worked):\n{src_text}\n")
                continue
            elapsed = time.time() - t0

            print(f"\nSources:\n{src_text}")
            print(f"({elapsed:.1f}s)")

            rating, reasons, notes = prompt_rating()

            turn_num += 1
            session["turns"].append({
                "turn": turn_num,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "question": question,
                "answer": body,
                "sources": sources,
                "latency_seconds": round(elapsed, 1),
                "rating": rating,
                "reasons": reasons,
                "notes": notes,
            })
            save_session(session, path)

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": body})
            last_chunks = chunks
            print()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        session["ended_at"] = datetime.now().isoformat(timespec="seconds")
        save_session(session, path)
        print(f"\nSaved {turn_num} question(s) to {path}. Goodbye, {researcher}!")


def _selfcheck():
    import tempfile
    assert slugify("Dr. Jane O'Brien!") == "dr_jane_o_brien"
    assert slugify("   ") == "researcher"

    prev = [{"id": "a"}, {"id": "b"}]
    fresh = [{"id": "b"}, {"id": "c"}, {"id": "d"}]
    merged = _merge_chunks(prev, fresh, cap=3)
    assert [c["id"] for c in merged] == ["a", "b", "c"], merged  # prev first, deduped, capped

    assert _merge_chunks([], fresh, cap=10) == fresh  # no previous turn -- fresh only
    assert _merge_chunks(prev, [], cap=10) == prev     # no fresh hits -- keep carrying prev

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = session_path("jane", datetime(2026, 7, 13, 15, 30, 0), base_dir=tmp_dir)
        assert path.endswith(os.path.join("jane", "20260713_153000.json"))
        save_session({"researcher": "jane", "turns": []}, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["researcher"] == "jane"
        assert not os.path.exists(path + ".tmp")  # atomic rename cleaned up the temp file
    print("ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        _selfcheck()
        sys.exit(0)
    main()

"""Chunk schema for the ARID RAG pipeline (KAN-9)."""

from dataclasses import dataclass, fields

VALID_LANGUAGES = {"cpp", "fcl", "python", "jsonnet"}  # fcl, not fhicl, idk man if we need to change it we can


@dataclass
class Chunk:
    id: str            # f"{file}_{symbol}_{start_line}"
    file: str          # relative path within repo
    start_line: int
    end_line: int
    symbol: str
    language: str      # one of VALID_LANGUAGES
    text: str


def make_chunk_id(file: str, symbol: str, start_line: int) -> str:
    return f"{file}_{symbol}_{start_line}"
# file is the repo-relative path, which keeps ids unique across the repo
# (repo alone was constant -> same symbol+line in two files collided).
# ex. CalibrateHits at line 42 of Calib/foo.cc -> Calib/foo.cc_CalibrateHits_42


def validate_chunk(chunk: dict) -> list[str]:
    """Return error strings; empty list means valid."""
    errors = [f"missing field: {f.name}" for f in fields(Chunk) if f.name not in chunk]
    if "language" in chunk and chunk["language"] not in VALID_LANGUAGES:
        errors.append(f"bad language: {chunk['language']!r}")
    if "start_line" in chunk and "end_line" in chunk and chunk["end_line"] < chunk["start_line"]:
        errors.append(f"end_line {chunk['end_line']} < start_line {chunk['start_line']}")
    return errors

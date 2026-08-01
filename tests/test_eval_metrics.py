"""Tests for EGEpipeline/eval/metrics.py's two pathology-specific metrics:
constructor_surfacing_rate (boilerplate ctors outranking the real-logic
method) and multiplicity_coverage (queries whose gold answer is more than one
chunk). Synthetic ranked-list/gold-set examples only -- pure functions, no
GPU/Qdrant/Ollama dependency, fully testable offline.
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")  # ponytail: Windows console is cp1252, needs utf-8
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "EGEpipeline", "eval"))
from metrics import (
    constructor_surfacing_rate,
    has_non_constructor_gold,
    is_constructor_chunk,
    multiplicity_coverage,
    multiplicity_coverage_summary,
    score_query,
)

# Real chunk ids lifted from EGEpipeline/eval/gold.jsonl, so the id-format
# assumptions (file_Namespace::Class::Symbol_line) are the real convention,
# not a made-up shape. Doubled "dunereco/dunereco/..." prefix (8/1): Tier-1
# extraction root sits one level above each repo clone, so dunereco's own
# internal LArSoft package dir (also named "dunereco") shows up twice -- see
# the 8/1 gold.jsonl id-fix commit message for the full explanation.
CTOR_ID = ("dunereco/dunereco/InfillChannels/art/InfillChannels_module.cc_"
           "Infill::InfillChannels::InfillChannels_83")
METHOD_ID = ("dunereco/dunereco/FDSelections/CCNuSelection_module.cc_"
             "FDSelection::CCNuSelection::RunHighestEnergyShowerSelection_1682")
TWO_PART_METHOD_ID = ("dunereco/dunereco/AnaUtils/DUNEAnaEventUtils.cxx_"
                       "DUNEAnaEventUtils::HasNeutrino_153")
FCL_ID = "dunereco/dunereco/CVN/adcutils/sp_fd_adcdump_job_example.fcl_services_12"
# synthetic 2-part (no namespace level) constructor, to check that path too
TWO_PART_CTOR_ID = "some/path/Foo.cc_Foo::Foo_10"

# Real multiplicity gold pair from gold.jsonl row 17 (two SelectTrack() tool
# implementations of the same interface -- both must surface).
TRACK_A = ("dunereco/dunereco/FDSelections/tools/LongestRecoVertexTrackSelector_tool.cc_"
           "FDSelectionTools::LongestRecoVertexTrackSelector::SelectTrack_15")
TRACK_B = ("dunereco/dunereco/FDSelections/tools/HighestPandizzleScoreRecoVertexTrackSelector_tool.cc_"
           "FDSelectionTools::HighestPandizzleScoreRecoVertexTrackSelector::SelectTrack_14")


# ---------------------------------------------------------------- constructor


def test_is_constructor_chunk_recognizes_real_constructor():
    assert is_constructor_chunk(CTOR_ID)


def test_is_constructor_chunk_recognizes_two_part_constructor():
    # no namespace level -- symbol is just "Class::Method", still a ctor
    assert is_constructor_chunk(TWO_PART_CTOR_ID)


def test_is_constructor_chunk_rejects_real_method():
    assert not is_constructor_chunk(METHOD_ID)


def test_is_constructor_chunk_rejects_two_part_method():
    # DUNEAnaEventUtils::HasNeutrino -- class name coincidentally also matches
    # the file's basename (DUNEAnaEventUtils.cxx), which is exactly the case
    # the file-vs-symbol split has to get right.
    assert not is_constructor_chunk(TWO_PART_METHOD_ID)


def test_is_constructor_chunk_false_for_non_cpp_chunk():
    # .fcl chunks have no '::' at all -- never a constructor
    assert not is_constructor_chunk(FCL_ID)


def test_has_non_constructor_gold():
    assert has_non_constructor_gold({METHOD_ID})
    assert not has_non_constructor_gold({CTOR_ID})
    assert has_non_constructor_gold({CTOR_ID, METHOD_ID})


def test_constructor_surfacing_rate_flags_ctor_beating_real_method():
    # top-1 is the boilerplate ctor, the real answer (METHOD_ID) is gold but
    # got outranked -- exactly the pathology the reranker/dedup work targets.
    ranked = [CTOR_ID, METHOD_ID, "some/other/Chunk.cc_Other::Thing_1"]
    gold = {METHOD_ID}
    assert constructor_surfacing_rate(ranked, gold, 1) == 1.0
    assert constructor_surfacing_rate(ranked, gold, 3) == 1.0 / 3.0


def test_constructor_surfacing_rate_zero_when_no_constructor_in_top_k():
    ranked = [METHOD_ID, "some/other/Chunk.cc_Other::Thing_1"]
    gold = {METHOD_ID}
    assert constructor_surfacing_rate(ranked, gold, 2) == 0.0


def test_constructor_surfacing_rate_zero_when_ctor_is_the_gold_answer():
    # ctor at top-1, but it IS the gold answer (e.g. InfillChannels reading
    # its own config) -- not the pathology, must not be penalized.
    ranked = [CTOR_ID, "some/other/Chunk.cc_Other::Thing_1"]
    gold = {CTOR_ID}
    assert constructor_surfacing_rate(ranked, gold, 1) == 0.0
    assert not has_non_constructor_gold(gold)  # confirms this query wouldn't count toward the "applicable" aggregate


def test_constructor_surfacing_rate_empty_ranked_list():
    assert constructor_surfacing_rate([], {METHOD_ID}, 5) == 0.0


def test_score_query_includes_constructor_surfacing():
    ranked = [CTOR_ID, METHOD_ID]
    gold = {METHOD_ID}
    scores = score_query(ranked, gold, ks=[1, 2])
    assert scores["constructor_surfacing"] == {1: 1.0, 2: 0.5}


# ---------------------------------------------------------------- multiplicity


def test_multiplicity_coverage_full():
    ranked = [TRACK_A, TRACK_B, "noise_1"]
    gold = {TRACK_A, TRACK_B}
    assert multiplicity_coverage(ranked, gold, 3) == 1.0


def test_multiplicity_coverage_partial():
    # only one of the two sibling implementations surfaces
    ranked = [TRACK_A, "noise_1", "noise_2"]
    gold = {TRACK_A, TRACK_B}
    assert multiplicity_coverage(ranked, gold, 3) == 0.5


def test_multiplicity_coverage_zero():
    ranked = ["noise_1", "noise_2"]
    gold = {TRACK_A, TRACK_B}
    assert multiplicity_coverage(ranked, gold, 2) == 0.0


def test_multiplicity_coverage_respects_k():
    # both siblings exist in the full ranked list, but only one is within top-1
    ranked = [TRACK_A, TRACK_B, "noise_1"]
    gold = {TRACK_A, TRACK_B}
    assert multiplicity_coverage(ranked, gold, 1) == 0.5
    assert multiplicity_coverage(ranked, gold, 2) == 1.0


def test_multiplicity_coverage_summary_breakdown():
    coverages = [1.0, 0.5, 0.0, 1.0, 0.3]
    summary = multiplicity_coverage_summary(coverages)
    assert summary == {"full": 2, "partial": 2, "zero": 1}


def test_multiplicity_coverage_summary_empty():
    assert multiplicity_coverage_summary([]) == {"full": 0, "partial": 0, "zero": 0}


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"{name} ok")
    print(f"all {len(tests)} eval-metrics tests passed")

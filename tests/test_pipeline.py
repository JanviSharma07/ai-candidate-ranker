"""
Basic sanity tests for the ranking pipeline.
Run with: pytest tests/
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import run_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATES_PATH = os.path.join(BASE_DIR, "data", "sample_candidates.json")

BACKEND_JD = (
    "Senior Backend Engineer (5+ years) needed. Must have Python, PostgreSQL, "
    "AWS, and Docker experience. Nice to have: Kubernetes, Kafka."
)


def test_pipeline_runs_end_to_end():
    result = run_pipeline(BACKEND_JD, CANDIDATES_PATH, top_k=5)
    assert "shortlist" in result
    assert len(result["shortlist"]) == 5
    assert result["total_candidates_evaluated"] > 5


def test_results_are_sorted_descending():
    result = run_pipeline(BACKEND_JD, CANDIDATES_PATH, top_k=10)
    scores = [r["final_score"] for r in result["shortlist"]]
    assert scores == sorted(scores, reverse=True)


def test_strong_match_outranks_irrelevant_profile():
    result = run_pipeline(BACKEND_JD, CANDIDATES_PATH, top_k=16)
    ranks_by_id = {r["candidate"]["id"]: r["rank"] for r in result["shortlist"]}
    # cand_001 = strong backend match, cand_014 = business analyst (irrelevant)
    assert ranks_by_id["cand_001"] < ranks_by_id["cand_014"]


def test_duplicate_summary_profiles_get_flagged():
    result = run_pipeline(BACKEND_JD, CANDIDATES_PATH, top_k=16)
    flagged = {
        r["candidate"]["id"]: r["quality_flags"]
        for r in result["shortlist"]
        if "duplicate_or_templated_summary" in r["quality_flags"]
    }
    assert "cand_007" in flagged and "cand_008" in flagged


def test_incomplete_profile_gets_quality_penalty():
    result = run_pipeline(BACKEND_JD, CANDIDATES_PATH, top_k=16)
    cand_006 = next(r for r in result["shortlist"] if r["candidate"]["id"] == "cand_006")
    assert cand_006["score_breakdown"]["quality_penalty_applied"] > 0


if __name__ == "__main__":
    test_pipeline_runs_end_to_end()
    test_results_are_sorted_descending()
    test_strong_match_outranks_irrelevant_profile()
    test_duplicate_summary_profiles_get_flagged()
    test_incomplete_profile_gets_quality_penalty()
    print("All tests passed.")

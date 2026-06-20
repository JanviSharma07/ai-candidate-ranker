"""
End-to-end pipeline: JD text -> ranked, explained candidate shortlist.
This is the single function both the API and the Streamlit UI call.
"""

from typing import Dict

from src.jd_parser import parse_job_description
from src.candidate_loader import load_candidates
from src.validation import validate_candidate
from src.scoring_engine import score_candidates
from src.explainability import explain_candidate


def run_pipeline(jd_text: str, candidates_path: str, top_k: int = 10) -> Dict:
    jd_requirements = parse_job_description(jd_text)
    candidates = load_candidates(candidates_path)

    all_summaries = [c["summary"] for c in candidates]
    quality_results = [validate_candidate(c, all_summaries) for c in candidates]

    ranked = score_candidates(jd_requirements, jd_text, candidates, quality_results)

    for r in ranked:
        r["explanation"] = explain_candidate(r)

    return {
        "jd_requirements": jd_requirements,
        "total_candidates_evaluated": len(candidates),
        "shortlist": ranked[:top_k],
    }

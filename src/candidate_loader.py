"""
Candidate Loader
-----------------
Loads candidate profiles from JSON and normalizes them into a consistent
internal representation so the rest of the pipeline never KeyErrors on
messy, incomplete, real-world-style profile data.
"""

import json
from typing import List, Dict


def load_candidates(path: str) -> List[Dict]:
    with open(path, "r") as f:
        candidates = json.load(f)
    return [_normalize(c) for c in candidates]


def _normalize(candidate: Dict) -> Dict:
    return {
        "id": candidate.get("id", "unknown"),
        "name": candidate.get("name", "Unnamed Candidate"),
        "skills": candidate.get("skills", []) or [],
        "experience_years": candidate.get("experience_years", 0) or 0,
        "summary": candidate.get("summary", "") or "",
        "education": candidate.get("education", "") or "",
        "last_active_days_ago": candidate.get("last_active_days_ago", 9999),
        "projects_count": candidate.get("projects_count", 0) or 0,
        "endorsements": candidate.get("endorsements", 0) or 0,
        "raw": candidate,
    }

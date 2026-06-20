"""
Profile Validation / Quality Layer
------------------------------------
Flags low-quality, incomplete, or suspicious candidate profiles BEFORE
ranking, so bad data doesn't silently distort the shortlist.

This module never deletes data -- it annotates a `quality_flags` list and a
`quality_penalty` (capped at 0.4) that the scoring engine applies as a
multiplicative discount. This keeps the system transparent and auditable
instead of silently dropping candidates a recruiter might want to see.
"""

from typing import Dict, List


def validate_candidate(candidate: Dict, all_summaries: List[str]) -> Dict:
    flags = []
    penalty = 0.0

    # 1. Missing critical fields
    if not candidate["skills"]:
        flags.append("no_skills_listed")
        penalty += 0.15
    if not candidate["summary"] or len(candidate["summary"].strip()) < 15:
        flags.append("insufficient_profile_detail")
        penalty += 0.1

    # 2. Implausible experience claims (e.g. <1 yr experience but 15+ skills listed)
    if 0 < candidate["experience_years"] < 1 and len(candidate["skills"]) > 15:
        flags.append("implausible_skill_to_experience_ratio")
        penalty += 0.15

    # 3. Duplicate / templated profile detection -- exact-match reused summaries
    #    across multiple "different" people is a common low-quality/spam signal.
    duplicate_count = all_summaries.count(candidate["summary"]) if candidate["summary"] else 0
    if duplicate_count > 1:
        flags.append("duplicate_or_templated_summary")
        penalty += 0.2

    # 4. Stale / inactive profile
    if candidate["last_active_days_ago"] > 365:
        flags.append("inactive_over_1_year")
        penalty += 0.1

    return {
        "quality_flags": flags,
        "quality_penalty": min(penalty, 0.4),  # cap so it never zeroes out a candidate
    }

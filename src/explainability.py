"""
Explainability Layer
-----------------------
Generates a human-readable justification for each candidate's rank, built
ENTIRELY from numbers already computed by the scoring engine
(score_breakdown, hard_requirement_details, quality_flags).

This is the key anti-hallucination design choice: the explanation generator
has no freedom to invent facts -- it can only restate values that already
exist in the pipeline's own output. There is no LLM call in the default
path, which means explanations are 100% grounded and fully reproducible.

`polish_with_llm()` is an optional add-on for nicer demo-video prose -- it
is still only allowed to REPHRASE the grounded facts; the prompt explicitly
forbids adding new claims.
"""

import os
from typing import Dict


def explain_candidate(result: Dict) -> str:
    breakdown = result["score_breakdown"]
    hard_details = result["hard_requirement_details"]
    flags = result["quality_flags"]

    lines = [
        f"Ranked #{result['rank']} with an overall fit score of {result['final_score']:.2f}/1.00."
    ]

    if hard_details["matched_must_have"]:
        lines.append(
            f"Matches {len(hard_details['matched_must_have'])} must-have skill(s): "
            f"{', '.join(hard_details['matched_must_have'])}."
        )
    if hard_details["missing_must_have"]:
        lines.append(
            f"Missing must-have skill(s): {', '.join(hard_details['missing_must_have'])}."
        )
    if hard_details["matched_nice_to_have"]:
        lines.append(
            f"Also brings nice-to-have skill(s): {', '.join(hard_details['matched_nice_to_have'])}."
        )
    if not hard_details["meets_experience_requirement"]:
        lines.append("Does not yet meet the minimum experience requirement for this role.")

    lines.append(
        f"Semantic profile-to-JD similarity score: {breakdown['semantic_fit']:.2f} "
        f"(captures relevant experience phrased differently than the JD)."
    )
    lines.append(
        f"Career-level alignment score: {breakdown['career_metadata_fit']:.2f}; "
        f"recent-activity/engagement score: {breakdown['behavioral_signal']:.2f}."
    )

    if flags:
        lines.append(
            f"Quality flags raised on this profile: {', '.join(flags)} "
            f"({breakdown['quality_penalty_applied']*100:.0f}% penalty applied to the final score)."
        )

    return " ".join(lines)


def polish_with_llm(grounded_explanation: str) -> str:
    """Optional: rephrase the grounded explanation more naturally using Claude.
    Strictly forbidden from adding any fact not already present in the input."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return grounded_explanation
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            system=(
                "Rewrite the following candidate evaluation in 2-3 fluent sentences "
                "for a recruiter. Do NOT add any fact, number, or skill that is not "
                "already present in the text. Do not soften or remove negative "
                "findings if present."
            ),
            messages=[{"role": "user", "content": grounded_explanation}],
        )
        return response.content[0].text.strip()
    except Exception:
        return grounded_explanation

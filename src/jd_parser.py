"""
JD Parser Module
-----------------
Extracts structured requirements from a raw job description.

Strategy:
1. If an ANTHROPIC_API_KEY is available, use Claude to extract a structured,
   strictly-grounded JSON of requirements (must-have skills, nice-to-have
   skills, min experience, seniority, role type, key responsibilities).
2. If no API key is available (offline / no-key demo mode), fall back to a
   deterministic rule-based extractor so the pipeline never breaks.

The LLM prompt is constrained to extract ONLY what is present in the text,
explicitly forbidding invented skills/requirements -- this directly supports
the "no hallucination" requirement from the challenge brief.
"""

import os
import json
import re
from typing import Optional

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


JD_EXTRACTION_SYSTEM_PROMPT = """You are a precise job description parser.
Extract ONLY information that is explicitly present in the job description text.
Do NOT invent, infer, or assume any skill, requirement, or detail that is not
directly stated or very strongly implied by the text.

Return ONLY valid JSON (no markdown, no commentary) matching this schema:
{
  "role_title": string,
  "seniority": "junior" | "mid" | "senior" | "lead" | "unspecified",
  "min_experience_years": number or null,
  "must_have_skills": [string],
  "nice_to_have_skills": [string],
  "responsibilities": [string],
  "soft_requirements": [string]
}

If a field cannot be determined from the text, use null or an empty list.
"""

# Extend this vocabulary to match the domains you expect to demo with.
SKILL_VOCAB = [
    "python", "java", "javascript", "typescript", "react", "node.js", "node",
    "sql", "postgresql", "mongodb", "aws", "azure", "gcp", "docker",
    "kubernetes", "machine learning", "deep learning", "nlp",
    "data science", "fastapi", "django", "flask", "next.js", "git",
    "rest api", "microservices", "ci/cd", "terraform", "pandas", "numpy",
    "tensorflow", "pytorch", "spark", "kafka", "redis", "graphql",
]


def _skill_present(skill: str, text_lower: str) -> bool:
    """Word-boundary-safe substring check, so 'sql' doesn't false-positive
    match inside 'postgresql'."""
    pattern = r"(?<![a-zA-Z])" + re.escape(skill) + r"(?![a-zA-Z])"
    return re.search(pattern, text_lower) is not None


def _rule_based_extract(jd_text: str) -> dict:
    """Deterministic fallback extractor - no external API needed."""
    text_lower = jd_text.lower()

    exp_match = re.search(r"(\d+)\+?\s*(?:years|yrs)", text_lower)
    min_exp = int(exp_match.group(1)) if exp_match else None

    seniority = "unspecified"
    for level, kws in [
        ("lead", ["lead", "principal", "staff"]),
        ("senior", ["senior", "sr."]),
        ("junior", ["junior", "entry level", "fresher"]),
    ]:
        if any(kw in text_lower for kw in kws):
            seniority = level
            break
    if seniority == "unspecified":
        seniority = "mid"

    found_skills = [s for s in SKILL_VOCAB if _skill_present(s, text_lower)]

    must_have, nice_to_have = found_skills, []
    split_markers = ["nice to have", "good to have", "preferred"]
    found_indices = [text_lower.find(m) for m in split_markers if m in text_lower]
    if found_indices:
        split_idx = min(found_indices)
        before, after = jd_text[:split_idx].lower(), jd_text[split_idx:].lower()
        must_have = [s for s in found_skills if s in before]
        nice_to_have = [s for s in found_skills if s in after and s not in must_have]

    role_match = re.search(r"^(.*?)(?:\n|$)", jd_text.strip())
    role_title = role_match.group(1)[:80] if role_match else "Unspecified Role"

    return {
        "role_title": role_title,
        "seniority": seniority,
        "min_experience_years": min_exp,
        "must_have_skills": must_have,
        "nice_to_have_skills": nice_to_have,
        "responsibilities": [],
        "soft_requirements": [],
        "_extraction_method": "rule_based_fallback",
    }


def _llm_extract(jd_text: str) -> Optional[dict]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not (_ANTHROPIC_AVAILABLE and api_key):
        return None
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=JD_EXTRACTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": jd_text}],
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        data["_extraction_method"] = "llm_claude"
        return data
    except Exception as e:
        print(f"[jd_parser] LLM extraction failed, falling back. Reason: {e}")
        return None


def parse_job_description(jd_text: str) -> dict:
    """Main entry point. Tries LLM extraction first, falls back to rules."""
    result = _llm_extract(jd_text)
    if result is None:
        result = _rule_based_extract(jd_text)
    return result

"""
Multi-Signal Scoring Engine
-----------------------------
Combines four independent signals into one explainable final score:

1. Semantic Fit          (40%) - embedding similarity between the JD and the
                                 candidate's summary/skills text. Catches
                                 synonyms / rephrased skills that exact
                                 keyword matching misses.
2. Hard Requirement Fit  (35%) - exact match against must-have / nice-to-have
                                 skills extracted from the JD, plus an
                                 experience-threshold check.
3. Career Metadata Fit   (15%) - seniority/experience-level alignment.
4. Behavioral Signal     (10%) - recency of activity + project/endorsement
                                 engagement, as a proxy for "is this profile
                                 alive and credible".

Each candidate's quality_penalty (from validation.py) is then applied as a
multiplicative discount on the combined score, so a flagged profile is
demoted rather than silently erased.

Embedding backend:
  - Primary: sentence-transformers (all-MiniLM-L6-v2) for genuine semantic
    similarity.
  - Fallback: TF-IDF + cosine similarity (scikit-learn) if the embedding
    model can't be downloaded (no internet / restricted environment). This
    means the demo NEVER crashes, which matters when judges run your code on
    an unfamiliar machine.
"""

import numpy as np
from typing import Dict, List, Tuple

_embedder = None
_embedder_backend = None


def _get_embedder():
    global _embedder, _embedder_backend
    if _embedder is not None:
        return _embedder, _embedder_backend
    try:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        _embedder_backend = "sentence-transformers"
    except Exception as e:
        print(f"[scoring_engine] sentence-transformers unavailable ({e}); using TF-IDF fallback.")
        _embedder = "tfidf"
        _embedder_backend = "tfidf"
    return _embedder, _embedder_backend


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def semantic_similarity(jd_text: str, candidate_texts: List[str]) -> List[float]:
    embedder, backend = _get_embedder()

    if backend == "sentence-transformers":
        jd_vec = embedder.encode([jd_text])
        cand_vecs = embedder.encode(candidate_texts)
        sims = _cosine_sim_matrix(jd_vec, cand_vecs)[0]
        return [float(x) for x in sims]

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    corpus = [jd_text] + candidate_texts
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    return [float(x) for x in sims]


def hard_requirement_score(jd_requirements: Dict, candidate: Dict) -> Tuple[float, Dict]:
    must_have = set(s.lower() for s in jd_requirements.get("must_have_skills", []) or [])
    nice_to_have = set(s.lower() for s in jd_requirements.get("nice_to_have_skills", []) or [])
    candidate_skills = set(s.lower() for s in candidate["skills"])

    matched_must = must_have & candidate_skills
    matched_nice = nice_to_have & candidate_skills

    must_ratio = len(matched_must) / len(must_have) if must_have else 1.0
    nice_ratio = len(matched_nice) / len(nice_to_have) if nice_to_have else 0.0

    min_exp = jd_requirements.get("min_experience_years")
    exp_ok = True
    if min_exp is not None:
        exp_ok = candidate["experience_years"] >= min_exp

    score = (0.75 * must_ratio) + (0.25 * nice_ratio)
    if min_exp is not None and not exp_ok:
        score *= 0.6  # penalize, don't zero out -- borderline candidates stay visible

    details = {
        "matched_must_have": sorted(matched_must),
        "missing_must_have": sorted(must_have - candidate_skills),
        "matched_nice_to_have": sorted(matched_nice),
        "meets_experience_requirement": exp_ok,
    }
    return min(score, 1.0), details


def career_metadata_score(jd_requirements: Dict, candidate: Dict) -> float:
    seniority_map = {"junior": 1, "mid": 3, "senior": 6, "lead": 9, "unspecified": 3}
    target_level = seniority_map.get(jd_requirements.get("seniority", "unspecified"), 3)
    gap = abs(candidate["experience_years"] - target_level)
    return round(max(0.0, 1.0 - (gap / 10.0)), 3)


def behavioral_score(candidate: Dict) -> float:
    recency_score = max(0.0, 1.0 - (candidate["last_active_days_ago"] / 365.0))
    engagement_score = min(1.0, (candidate["projects_count"] * 0.1) + (candidate["endorsements"] * 0.02))
    return round((0.6 * recency_score) + (0.4 * engagement_score), 3)


def score_candidates(jd_requirements: Dict, jd_text: str, candidates: List[Dict],
                      quality_results: List[Dict]) -> List[Dict]:
    candidate_texts = [
        f"{c['summary']} Skills: {', '.join(c['skills'])}. Education: {c['education']}."
        for c in candidates
    ]
    semantic_scores = semantic_similarity(jd_text, candidate_texts)

    results = []
    for candidate, sem_score, quality in zip(candidates, semantic_scores, quality_results):
        hard_score, hard_details = hard_requirement_score(jd_requirements, candidate)
        meta_score = career_metadata_score(jd_requirements, candidate)
        behav_score = behavioral_score(candidate)

        combined = (
            0.40 * max(sem_score, 0.0)
            + 0.35 * hard_score
            + 0.15 * meta_score
            + 0.10 * behav_score
        )
        final_score = combined * (1.0 - quality["quality_penalty"])

        results.append({
            "candidate": candidate,
            "final_score": round(final_score, 4),
            "score_breakdown": {
                "semantic_fit": round(sem_score, 4),
                "hard_requirement_fit": round(hard_score, 4),
                "career_metadata_fit": meta_score,
                "behavioral_signal": behav_score,
                "quality_penalty_applied": quality["quality_penalty"],
            },
            "hard_requirement_details": hard_details,
            "quality_flags": quality["quality_flags"],
        })

    results.sort(key=lambda r: r["final_score"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank
    return results

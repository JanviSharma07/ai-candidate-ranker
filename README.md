# Intelligent Candidate Discovery

Built for **Hack2Skill — India RUNS: Data & AI Challenge**.

A Proof of Concept that goes beyond keyword filtering to *intelligently rank*
candidates against a job description — combining semantic understanding,
hard-requirement matching, career metadata, and behavioral signals into one
explainable fit score.

## What it does

1. **Parses the JD** into structured requirements (must-have skills,
   nice-to-have skills, seniority, minimum experience) — using Claude if an
   API key is configured, or a deterministic rule-based extractor if not.
2. **Validates every candidate profile** for missing fields, duplicate/templated
   content, implausible claims, and staleness — flagging issues instead of
   silently dropping candidates.
3. **Scores every candidate** on four independent signals:
   - Semantic fit (embedding similarity — catches relevant experience phrased
     differently than the JD)
   - Hard requirement fit (must-have / nice-to-have skill match + experience
     threshold)
   - Career metadata fit (seniority/experience alignment)
   - Behavioral signal (activity recency + engagement)
4. **Ranks candidates** by a weighted combination of the above, discounted by
   any quality penalty.
5. **Explains every ranking** using only the numbers already computed —
   no LLM is required to generate the explanation, so it can never hallucinate
   a justification that isn't backed by the data.

## Architecture

```
                 ┌──────────────────┐
   JD text  ───► │   JD Parser       │  (LLM or rule-based fallback)
                 └────────┬──────────┘
                          │  structured requirements
                          ▼
┌─────────────┐   ┌──────────────────┐
│  Candidate  │──►│  Validation /     │  flags + quality penalty
│  Profiles   │   │  Quality Layer    │
└─────────────┘   └────────┬──────────┘
                          ▼
                 ┌──────────────────┐
                 │ Scoring Engine    │  semantic + hard-match +
                 │ (4 signals)       │  metadata + behavioral
                 └────────┬──────────┘
                          ▼
                 ┌──────────────────┐
                 │ Explainability    │  grounded, fact-only justification
                 │ Layer             │
                 └────────┬──────────┘
                          ▼
                 Ranked, explained shortlist
```

## Project structure

```
ai-candidate-ranker/
├── api/main.py             FastAPI service (POST /rank)
├── app.py                  Streamlit demo UI
├── src/
│   ├── jd_parser.py        JD → structured requirements
│   ├── candidate_loader.py Loads + normalizes candidate JSON
│   ├── validation.py       Quality flags + penalty scoring
│   ├── scoring_engine.py   4-signal scoring + ranking
│   ├── explainability.py   Grounded explanation generator
│   └── pipeline.py         Orchestrates the full flow
├── data/
│   ├── sample_jds.json
│   └── sample_candidates.json
└── tests/test_pipeline.py
```

## Setup

```bash
git clone <your-repo-url>
cd ai-candidate-ranker
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # optional: add ANTHROPIC_API_KEY for LLM-based JD parsing
```

## Running it

**Demo UI (recommended for judging/demo video):**
```bash
streamlit run app.py
```

**API:**
```bash
uvicorn api.main:app --reload --port 8000
# then POST to http://localhost:8000/rank with {"jd_text": "...", "top_k": 5}
# interactive docs at http://localhost:8000/docs
```

**Tests:**
```bash
pytest tests/ -v
```

## Why this design

- **No hallucination, by construction.** Explanations are template-assembled
  from the scoring engine's own numbers — there is no step where free-form
  text generation could introduce an unsupported claim. An optional LLM
  "polish" pass (`explainability.polish_with_llm`) is allowed to *rephrase*
  but never to add new facts.
- **Never crashes from missing dependencies.** The semantic similarity
  module tries `sentence-transformers` first and silently falls back to a
  TF-IDF + cosine-similarity baseline if the embedding model can't be
  downloaded — so the system runs even fully offline.
- **Data quality is surfaced, not hidden.** Suspicious, duplicate, or
  incomplete profiles are flagged and down-weighted rather than excluded,
  keeping the recruiter in control.
- **Runtime/compute friendly.** Default semantic backend is a lightweight
  22M-parameter sentence embedding model; the rule-based JD parser and
  TF-IDF fallback have no external dependency at all, keeping the system
  usable within tight compute budgets.

## Possible extensions

- Swap the synthetic dataset for a real resume corpus (e.g. parse PDFs with
  a resume-parsing library) and real activity data from a recruiting
  platform's API.
- Add a learned re-ranker (e.g. gradient-boosted model) trained on
  recruiter feedback to replace the fixed signal weights.
- Add a vector database (e.g. FAISS/pgvector) for retrieval at scale before
  scoring, instead of scoring every candidate linearly.

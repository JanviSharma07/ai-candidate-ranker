"""
FastAPI service exposing the ranking pipeline.

Run with: uvicorn api.main:app --reload --port 8000
Then open http://localhost:8000/docs for interactive API docs.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline import run_pipeline

app = FastAPI(title="Intelligent Candidate Discovery API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class RankRequest(BaseModel):
    jd_text: str
    top_k: int = 10


@app.get("/")
def health():
    return {"status": "ok", "service": "intelligent-candidate-discovery"}


@app.post("/rank")
def rank(request: RankRequest):
    result = run_pipeline(
        jd_text=request.jd_text,
        candidates_path=os.path.join(DATA_DIR, "sample_candidates.json"),
        top_k=request.top_k,
    )
    return result

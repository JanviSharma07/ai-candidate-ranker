"""
Streamlit demo UI for the Intelligent Candidate Discovery system.

Run with: streamlit run app.py
"""

import json
import os

import streamlit as st

from src.pipeline import run_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "sample_candidates.json")
JDS_PATH = os.path.join(BASE_DIR, "data", "sample_jds.json")

st.set_page_config(page_title="Intelligent Candidate Discovery", layout="wide")
st.title("Intelligent Candidate Discovery")
st.caption("Paste a job description and get a ranked, explained shortlist.")

with open(JDS_PATH) as f:
    sample_jds = json.load(f)

jd_choice = st.selectbox(
    "Use a sample JD or write your own:",
    ["-- write my own --"] + [jd["title"] for jd in sample_jds],
)

if jd_choice == "-- write my own --":
    jd_text = st.text_area("Job Description", height=220)
else:
    jd_text = next(jd["text"] for jd in sample_jds if jd["title"] == jd_choice)
    st.text_area("Job Description", value=jd_text, height=220, key="prefilled_jd")

top_k = st.slider("Shortlist size", 3, 15, 5)

if st.button("Rank Candidates", type="primary"):
    if not jd_text.strip():
        st.warning("Please enter a job description.")
    else:
        with st.spinner("Parsing JD, scoring candidates, generating explanations..."):
            result = run_pipeline(jd_text, DATA_PATH, top_k=top_k)

        st.subheader("Extracted JD Requirements")
        st.json(result["jd_requirements"])

        st.subheader(f"Top {top_k} of {result['total_candidates_evaluated']} candidates evaluated")
        for r in result["shortlist"]:
            c = r["candidate"]
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**#{r['rank']} \u2014 {c['name']}**")
                    st.write(r["explanation"])
                    if r["quality_flags"]:
                        st.warning(f"Quality flags: {', '.join(r['quality_flags'])}")
                with col2:
                    st.metric("Fit Score", f"{r['final_score']:.2f}")
                st.caption(
                    f"Semantic: {r['score_breakdown']['semantic_fit']:.2f} | "
                    f"Hard-match: {r['score_breakdown']['hard_requirement_fit']:.2f} | "
                    f"Career: {r['score_breakdown']['career_metadata_fit']:.2f} | "
                    f"Behavioral: {r['score_breakdown']['behavioral_signal']:.2f}"
                )

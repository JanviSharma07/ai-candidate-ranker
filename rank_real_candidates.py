import json, csv
from src.scoring_engine import semantic_similarity

JD_TEXT = """Senior AI Engineer - Founding Team at Redrob AI.
Must have: production experience with embeddings-based retrieval systems
sentence-transformers OpenAI embeddings BGE E5 vector databases hybrid search
Pinecone Weaviate Qdrant Milvus FAISS Elasticsearch strong Python evaluation
frameworks ranking systems NDCG MRR MAP LLM fine-tuning LoRA QLoRA
learning-to-rank models HR-tech marketplace distributed systems. 5-9 years."""

BATCH = 200
TOP_K = 100

def build_text(c):
    profile = c.get("profile", {})
    skills = c.get("skills", [])
    skill_names = [s.get("name","") if isinstance(s,dict) else s for s in skills[:15]]
    history = c.get("career_history", [])
    desc = " ".join(h.get("description","") for h in history[:1])
    return (profile.get("headline","") + " " + profile.get("summary","")[:300] + " " + " ".join(skill_names) + " " + desc[:200])[:600]

def score_one(c, sim):
    profile = c.get("profile", {})
    yoe = float(profile.get("years_of_experience", 0) or 0)
    gap = abs(yoe - 7)
    career = max(0.0, 1.0 - gap/10.0)
    loc = (profile.get("location","") + profile.get("country","")).lower()
    loc_bonus = 0.05 if any(x in loc for x in ["pune","noida","india","delhi","mumbai","bangalore","hyderabad"]) else 0.0
    signals = c.get("redrob_signals", {}) or {}
    recency = float(signals.get("days_since_last_active", 999) or 999)
    rec_score = max(0.0, 1.0 - recency/365.0)
    rr = float(signals.get("response_rate", 0) or 0)
    behavioral = 0.6*rec_score + 0.4*rr
    return round(0.50*max(sim,0.0) + 0.25*career + 0.15*behavioral + 0.10*loc_bonus, 4)

def make_reasoning(c, score):
    p = c.get("profile", {})
    skills = c.get("skills", [])
    top = [s.get("name","") if isinstance(s,dict) else s for s in skills[:3]]
    rr = float((c.get("redrob_signals") or {}).get("response_rate", 0) or 0)
    return "{} with {:.1f} yrs exp; top skills: {}; response rate {:.2f}.".format(
        p.get("current_title","Unknown"), float(p.get("years_of_experience",0) or 0), ", ".join(top), rr)

print("Loading and scoring 100k candidates in batches...")
top_results = []

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:
    batch = []
    count = 0
    for line in f:
        line = line.strip()
        if not line:
            continue
        batch.append(json.loads(line))
        count += 1
        if len(batch) == BATCH:
            texts = [build_text(c) for c in batch]
            sims = semantic_similarity(JD_TEXT, texts)
            for c, sim in zip(batch, sims):
                sc = score_one(c, sim)
                top_results.append((sc, c["candidate_id"], c))
            top_results.sort(key=lambda x: x[0], reverse=True)
            top_results = top_results[:500]
            batch = []
            if count % 10000 == 0:
                print(str(count) + " processed, current best score: " + str(top_results[0][0]))
    if batch:
        texts = [build_text(c) for c in batch]
        sims = semantic_similarity(JD_TEXT, texts)
        for c, sim in zip(batch, sims):
            sc = score_one(c, sim)
            top_results.append((sc, c["candidate_id"], c))
        top_results.sort(key=lambda x: x[0], reverse=True)
        top_results = top_results[:500]

print("Writing submission.csv...")
with open("submission.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for rank, (score, cid, c) in enumerate(top_results[:TOP_K], start=1):
        writer.writerow([cid, rank, score, make_reasoning(c, score)])

print("Done! Top 5:")
for rank, (score, cid, c) in enumerate(top_results[:5], start=1):
    print(str(rank) + ". " + cid + " score=" + str(score) + " " + c["profile"].get("current_title",""))
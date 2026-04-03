import os
import sys
import json
import re
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "model_cache")

os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME, cache_folder=CACHE_DIR)

ROLE_PROFILES = {
    "Full Stack Developer": "Builds frontend and backend web applications using React, Node.js, JavaScript, HTML, CSS, APIs, and databases.",
    "Backend Developer": "Builds server-side systems, APIs, authentication, databases, and backend services using Node.js, Express, SQL, MySQL, PostgreSQL, and REST APIs.",
    "Frontend Developer": "Builds responsive user interfaces using React, JavaScript, TypeScript, HTML, CSS, and modern frontend frameworks.",
    "Machine Learning Engineer": "Designs and deploys machine learning solutions using Python, NumPy, Pandas, scikit-learn, deep learning, NLP, and model deployment.",
    "Data Scientist": "Analyzes data, builds predictive models, and performs statistical analysis using Python, Pandas, NumPy, SQL, and machine learning.",
    "Software Engineer": "Builds software systems with strong programming, OOP, DSA, problem solving, Java, C++, Python, Git, and SQL."
}

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def split_resume_lines(text: str):
    lines = []
    for raw in text.split("\n"):
        line = clean_text(raw)
        if len(line) >= 20:
            lines.append(line)

    if not lines:
        chunks = re.split(r"[.!?]", text)
        lines = [clean_text(c) for c in chunks if len(clean_text(c)) >= 20]

    return lines

def semantic_role_prediction(text):
    if not text:
        return {"role": "General Software Role", "confidence": 0.0, "topRoles": []}

    text_emb = model.encode([text], normalize_embeddings=True)

    best_role = "General Software Role"
    best_score = 0.0
    all_scores = []

    for role, desc in ROLE_PROFILES.items():
        role_emb = model.encode([desc], normalize_embeddings=True)
        score = float(cosine_similarity(text_emb, role_emb)[0][0]) * 100
        all_scores.append({"role": role, "score": round(score, 2)})

        if score > best_score:
            best_score = score
            best_role = role

    all_scores.sort(key=lambda x: x["score"], reverse=True)

    return {
        "role": best_role,
        "confidence": round(max(0.0, min(100.0, best_score)), 2),
        "topRoles": all_scores[:3]
    }

def analyze(resume_text, job_description):
    resume_text = clean_text(resume_text)
    job_description = clean_text(job_description)

    if not resume_text or not job_description:
        return {
            "semanticScore": 0.0,
            "topMatchedLines": [],
            "model": MODEL_NAME,
            "rolePrediction": {
                "resume": semantic_role_prediction(resume_text),
                "job": semantic_role_prediction(job_description)
            }
        }

    resume_emb = model.encode([resume_text], normalize_embeddings=True)
    job_emb = model.encode([job_description], normalize_embeddings=True)

    semantic_score = float(cosine_similarity(resume_emb, job_emb)[0][0]) * 100
    semantic_score = max(0.0, min(100.0, semantic_score))

    lines = split_resume_lines(resume_text)
    top_matched_lines = []

    if lines:
        line_emb = model.encode(lines, normalize_embeddings=True)
        scores = cosine_similarity(line_emb, job_emb).ravel()
        top_idx = np.argsort(scores)[::-1][:5]
        top_matched_lines = [lines[i] for i in top_idx]

    return {
        "semanticScore": round(semantic_score, 2),
        "topMatchedLines": top_matched_lines,
        "model": MODEL_NAME,
        "rolePrediction": {
            "resume": semantic_role_prediction(resume_text),
            "job": semantic_role_prediction(job_description)
        }
    }

if __name__ == "__main__":
    payload = json.loads(sys.stdin.read())
    result = analyze(
        payload.get("resume_text", ""),
        payload.get("job_description", "")
    )
    print(json.dumps(result))
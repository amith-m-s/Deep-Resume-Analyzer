import os
import sys
import json
import re
import warnings
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "model_cache")

os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

warnings.filterwarnings("ignore")

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Faster model for local + deploy use
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"
model = SentenceTransformer(MODEL_NAME, cache_folder=CACHE_DIR)

ROLE_PROFILES = {
    "Full Stack Developer": "Builds frontend and backend web applications using React, Node.js, JavaScript, HTML, CSS, APIs, and databases.",
    "Backend Developer": "Builds server-side systems, APIs, authentication, databases, and backend services using Node.js, Express, SQL.",
    "Frontend Developer": "Builds UI using React, JavaScript, HTML, CSS.",
    "Machine Learning Engineer": "Python, ML, deep learning, NLP.",
    "Data Scientist": "Python, Pandas, NumPy, ML, statistics.",
    "Software Engineer": "Java, C++, OOP, DSA."
}

ROLE_EMB = {
    role: model.encode([desc], normalize_embeddings=True)
    for role, desc in ROLE_PROFILES.items()
}

def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()

def split_resume_lines(text):
    lines = []
    for raw in text.split("\n"):
        line = clean(raw)
        if len(line) >= 20:
            lines.append(line)

    if not lines:
        chunks = re.split(r"[.!?]", text)
        lines = [clean(c) for c in chunks if len(clean(c)) >= 20]

    return lines

def semantic_role_prediction(text):
    if not text:
        return {"role": "General Software Role", "confidence": 0.0, "topRoles": []}

    text_emb = model.encode([text], normalize_embeddings=True)

    scores = []
    best_role = "General Software Role"
    best_score = 0.0

    for role, emb in ROLE_EMB.items():
        score = float(cosine_similarity(text_emb, emb)[0][0]) * 100
        scores.append({"role": role, "score": round(score, 2)})
        if score > best_score:
            best_score = score
            best_role = role

    scores.sort(key=lambda x: x["score"], reverse=True)

    return {
        "role": best_role,
        "confidence": round(max(0.0, min(100.0, best_score)), 2),
        "topRoles": scores[:3]
    }

def analyze(resume_text, job_description):
    resume_text = clean(resume_text)
    job_description = clean(job_description)

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

def main():
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue

        try:
            payload = json.loads(line)
            result = analyze(
                payload.get("resume_text", ""),
                payload.get("job_description", "")
            )
            output = {
                "id": payload.get("id"),
                "ok": True,
                "result": result
            }
        except Exception as e:
            output = {
                "id": payload.get("id") if "payload" in locals() else None,
                "ok": False,
                "error": str(e)
            }

        sys.stdout.write(json.dumps(output) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
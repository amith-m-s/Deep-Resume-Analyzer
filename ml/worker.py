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

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"
model = SentenceTransformer(MODEL_NAME, cache_folder=CACHE_DIR)

ROLE_PROFILES = {
    "Software Engineer": {
        "desc": "Builds software systems with strong programming, OOP, DSA, problem solving, Python, C#, Java, C++, Git, SQL.",
        "keywords": ["python", "c#", "java", "c++", "sql", "oop", "dsa", "backend", "software"]
    },
    "Backend Developer": {
        "desc": "Builds APIs, server-side logic, databases, authentication, Node.js, Express, SQL, MySQL, PostgreSQL.",
        "keywords": ["node.js", "node", "express", "rest api", "sql", "mysql", "postgresql", "backend", "api"]
    },
    "Full Stack Developer": {
        "desc": "Builds frontend and backend web applications using React, Node.js, JavaScript, HTML, CSS, APIs, and databases.",
        "keywords": ["react", "node.js", "node", "express", "javascript", "html", "css", "sql", "full stack"]
    },
    "Frontend Developer": {
        "desc": "Builds responsive user interfaces using React, JavaScript, TypeScript, HTML, CSS.",
        "keywords": ["react", "javascript", "typescript", "html", "css", "frontend"]
    },
    "Data Science Intern": {
        "desc": "Works on data preprocessing, exploratory data analysis, statistics, probability, Pandas, NumPy, Matplotlib, model building.",
        "keywords": ["data science", "eda", "exploratory data analysis", "pandas", "numpy", "matplotlib", "statistics", "probability", "model building", "data preprocessing"]
    },
    "Machine Learning Intern": {
        "desc": "Builds predictive models, trains machine learning systems, evaluates models, Python, ML, deep learning, NLP.",
        "keywords": ["machine learning", "ml", "deep learning", "predictive", "model", "training", "evaluation", "python"]
    },
    "Data Analyst": {
        "desc": "Analyzes datasets using SQL, Excel, Pandas, NumPy, visualization, statistics, dashboards, business insights.",
        "keywords": ["sql", "pandas", "numpy", "visualization", "statistics", "dashboard", "data analyst"]
    }
}

ROLE_EMB = {
    role: model.encode([meta["desc"]], normalize_embeddings=True)
    for role, meta in ROLE_PROFILES.items()
}

SECTION_WEIGHTS = {
    "professional summary": 1.8,
    "summary": 1.8,
    "technical skills": 2.0,
    "skills": 1.9,
    "projects": 1.7,
    "experience": 1.5,
    "work experience": 1.5,
    "internship": 1.4,
    "certifications": 0.5,
    "education": 0.4,
    "interests": 0.2,
    "additional information": 0.3,
    "default": 1.0
}

SECTION_HEADINGS = [
    ("professional summary", re.compile(r"^\s*professional summary\s*$", re.I)),
    ("summary", re.compile(r"^\s*summary\s*$", re.I)),
    ("technical skills", re.compile(r"^\s*technical skills\s*$", re.I)),
    ("skills", re.compile(r"^\s*skills\b.*$", re.I)),
    ("projects", re.compile(r"^\s*projects\s*$", re.I)),
    ("experience", re.compile(r"^\s*experience\s*$", re.I)),
    ("work experience", re.compile(r"^\s*work experience\s*$", re.I)),
    ("internship", re.compile(r"^\s*internship\s*$", re.I)),
    ("certifications", re.compile(r"^\s*certifications?\s*$", re.I)),
    ("education", re.compile(r"^\s*education\s*$", re.I)),
    ("interests", re.compile(r"^\s*interests?\s*$", re.I)),
    ("additional information", re.compile(r"^\s*additional information\s*$", re.I)),
]

def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()

def section_weight(section):
    return SECTION_WEIGHTS.get(section, SECTION_WEIGHTS["default"])

def split_resume_lines_with_sections(text):
    lines = []
    current_section = "default"

    for raw in text.split("\n"):
        stripped = raw.strip()
        if not stripped:
            continue

        found_heading = False
        for section_name, pattern in SECTION_HEADINGS:
            if pattern.match(stripped):
                current_section = section_name
                found_heading = True
                break

        if found_heading:
            continue

        line = clean(raw)
        if len(line) >= 12:
            lines.append((line, current_section))

    if not lines:
        chunks = re.split(r"[.!?]", text)
        lines = [(clean(c), "default") for c in chunks if len(clean(c)) >= 12]

    return lines

def semantic_role_prediction(text):
    if not text:
        return {"role": "General Software Role", "confidence": 0.0, "topRoles": []}

    text_emb = model.encode([text], normalize_embeddings=True)

    scores = []
    best_role = "General Software Role"
    best_score = 0.0

    weighted_lines = split_resume_lines_with_sections(text)

    for role, meta in ROLE_PROFILES.items():
        role_emb = ROLE_EMB[role]
        semantic_score = float(cosine_similarity(text_emb, role_emb)[0][0]) * 100

        keyword_points = 0.0
        lowered_text = text.lower()

        for kw in meta["keywords"]:
            kw_l = kw.lower()
            for line, sec in weighted_lines:
                if kw_l in line.lower():
                    keyword_points += section_weight(sec)

        # Stronger sections influence the score more; certifications/interests are weak.
        final_score = semantic_score * 0.58 + min(40.0, keyword_points * 5.0)

        scores.append({
            "role": role,
            "score": round(final_score, 2)
        })

        if final_score > best_score:
            best_score = final_score
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

    lines = split_resume_lines_with_sections(resume_text)
    top_matched_lines = []

    if lines:
        plain_lines = [x[0] for x in lines]
        line_emb = model.encode(plain_lines, normalize_embeddings=True)
        scores = cosine_similarity(line_emb, job_emb).ravel()
        top_idx = np.argsort(scores)[::-1][:5]
        top_matched_lines = [plain_lines[i] for i in top_idx]

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
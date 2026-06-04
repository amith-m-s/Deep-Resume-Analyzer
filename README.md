# Deep Resume Analyzer

**Semantic resume-to-JD matching system. 3-service Docker architecture. Live in production.**

[![Live Demo](https://img.shields.io/badge/demo-live-22c55e?style=flat-square&logo=vercel&logoColor=white)](https://deep-resume-analyzer.vercel.app)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://github.com/amith-m-s/Deep-Resume-Analyzer)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://github.com/amith-m-s/Deep-Resume-Analyzer)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://github.com/amith-m-s/Deep-Resume-Analyzer)

---

## What This Is

Traditional ATS systems fail because they rely purely on keyword matching — a data science resume can score 70%+ against a frontend JD just on semantic overlap. This system solves that with a hybrid model that combines semantic understanding, keyword precision, and domain skill validation.

Three Docker services. End-to-end PDF processing under 3 seconds. Deployed on Vercel with CDN-backed static delivery.

---

## Architecture

```
┌──────────────────┐         HTTP/REST          ┌────────────────────────┐
│  React Frontend  │ ────────────────────────▶  │  Express.js Gateway    │
│  Vercel CDN      │                             │  JWT Auth              │
│  /client         │ ◀────────────────────────  │  Rate Limiting         │
└──────────────────┘                             │  Multer (PDF upload)   │
                                                 └────────────┬───────────┘
                                                              │ REST
                                                              ▼
                                                 ┌────────────────────────┐
                                                 │  FastAPI NLP Service   │
                                                 │  /ml                   │
                                                 │                        │
                                                 │  normalize             │
                                                 │    → tokenize          │
                                                 │      → embed           │
                                                 │        → score         │
                                                 │          → rank        │
                                                 │            → gap-detect│
                                                 └────────────────────────┘
```

**Docker Compose wires all three services** with environment parity between development and production. The NLP service runs in its own container — independently scalable without touching the gateway or frontend.

---

## The Scoring Model

```python
final_score = (
    0.60 * cosine_similarity   # semantic context match
  + 0.25 * keyword_precision   # exact ATS term matching
  + 0.15 * domain_skill_score  # validated skill overlap
)
```

**Why these weights, not equal thirds?**

Cosine similarity captures context but overfits semantics. Without the keyword penalty, a resume using "constructed distributed data pipelines" against a JD saying "built ETL jobs" scores high semantically but would fail every real ATS filter. The 0.25 keyword weight forces precision. Domain skill scoring at 0.15 acts as a binary gate — missing critical skills should penalize the score but not dominate it.

**Why the pipeline is designed this way:**

Each stage (`normalize → tokenize → embed → score`) has a defined input/output contract. Swapping the current embedding model for BERT or a fine-tuned sentence transformer requires changing exactly one layer. The scoring and ranking logic stay untouched.

---

## Project Structure

```
Deep-Resume-Analyzer/
├── client/              # React frontend — Tailwind, PDF upload, results UI
├── ml/                  # FastAPI NLP microservice
│   ├── analyzer.py      # Core NLP pipeline
│   ├── scorer.py        # Hybrid scoring model
│   └── requirements.txt
├── server.js            # Express.js API gateway
├── package.json
├── Dockerfile
├── requirements.txt
└── docker-compose.yml   # Wires all 3 services with health checks
```

---

## Running Locally

**With Docker (recommended):**
```bash
git clone https://github.com/amith-m-s/Deep-Resume-Analyzer
cd Deep-Resume-Analyzer
docker compose up --build
```

Frontend: `http://localhost:3000`
Gateway: `http://localhost:5000`
NLP service: `http://localhost:8000`

**Without Docker:**
```bash
# Terminal 1 — NLP service
cd ml
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 2 — Express gateway
npm install
node server.js

# Terminal 3 — React frontend
cd client
npm install && npm start
```

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | React · Tailwind CSS · Vercel CDN |
| API Gateway | Node.js · Express.js · JWT Auth · Multer · Rate Limiting |
| NLP Service | Python · FastAPI · Sentence Embeddings · Cosine Similarity |
| Infrastructure | Docker · Docker Compose · GitHub Actions |

---

## Honest Limitations

- The current embedding model is a lightweight sentence encoder — not BERT. Accuracy on domain-specific JDs (medical, legal) degrades without domain-specific fine-tuning.
- No persistent database — results are not stored between sessions.
- No multi-resume batch processing yet.
- Test coverage is thin. The scoring model has been manually validated against test cases but lacks an automated pytest suite.

**Planned:**
- BERT/transformer embedding integration (pipeline is already staged for this)
- PostgreSQL persistence for result history
- Automated test suite with controlled resume/JD pairs

---

## Live Demo

[deep-resume-analyzer.vercel.app](https://deep-resume-analyzer.vercel.app)

Upload any resume PDF and a job description. The system returns a match score, matched skills, missing skills, and a ranked recommendation.

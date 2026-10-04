# Deep Resume Analyzer

**Semantic resume-to-job-description matching application with a React frontend, Express gateway, and Python NLP inference worker.**

## What it is

The application extracts text from uploaded PDF resumes, generates sentence embeddings with a lightweight MiniLM model, computes semantic similarity, performs rule-based skill extraction, predicts likely roles, and produces a hybrid match score.

### Current architecture

~~~
React client
   |
   v
Express.js gateway
   |
   +--> PDF extraction
   +--> keyword/skill matching
   +--> rate limiting
   |
   v
Python inference worker
   |
   +--> SentenceTransformer
   +--> cosine similarity
   +--> role prediction
   +--> matched-line ranking
~~~

The current repository **does not contain a separate FastAPI NLP microservice or a committed docker-compose.yml**. The Python model runs as an inference worker process launched by the Node application in the Docker image.

## Hybrid scoring

~~~
final_score =
    0.60 * semantic_similarity
  + 0.25 * keyword_precision
  + 0.15 * domain_skill_score
~~~

These weights are an engineering design choice. They should not be interpreted as a statistically validated claim of reduced false positives until benchmark results are published.

## Model

Current inference model:

sentence-transformers/all-MiniLM-L6-v2

Role prediction and skill extraction also use deterministic rules/role profiles to complement the embedding score.

## Run locally

~~~
git clone https://github.com/amith-m-s/Deep-Resume-Analyzer
cd Deep-Resume-Analyzer
npm install
node server.js
~~~

The React client lives under client/ and can be started independently with its own npm script.

## Honest limitations

- No persistent result database.
- Test coverage is still limited.
- Domain-specific hiring accuracy has not been benchmarked against a labeled dataset.
- The live demo is intended for demonstration, not real hiring decisions.
- Batch processing and persistent candidate history are future work.

## Live demo

https://deep-resume-analyzer.vercel.app

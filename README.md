# 🧠 Deep Resume Analyzer

A system for **semantic resume-job matching** using embeddings, ATS-style scoring, and skill gap analysis.

Built to simulate **real-world hiring pipelines**, not just keyword matching.

---

## 🚀 Overview

This project analyzes resumes against job descriptions by going beyond simple keyword checks.

It uses **semantic similarity, structured scoring, and skill extraction** to evaluate how well a candidate fits a role.

---

## ⚙️ Core Features

* Semantic similarity using embeddings
* ATS-style scoring system
* Keyword and skill extraction
* Skill gap detection
* Structured evaluation pipeline

---

## 🏗️ Architecture

```
            ┌──────────────┐
            │   Resume     │
            └──────┬───────┘
                   │
                   ▼
         ┌──────────────────┐
         │ Text Processing  │
         └────────┬─────────┘
                  │
                  ▼
        ┌────────────────────┐
        │ Embedding Engine   │
        └────────┬───────────┘
                 │
                 ▼
      ┌────────────────────────┐
      │ Similarity Computation │
      └────────┬───────────────┘
               │
               ▼
     ┌──────────────────────────┐
     │ ATS Scoring + Analysis   │
     └────────┬─────────────────┘
              │
              ▼
      ┌──────────────────────┐
      │ Final Evaluation     │
      └──────────────────────┘
```

---

## 🧠 How It Works

### 1. Text Processing

* Cleans and normalizes resume + job description
* Removes noise and extracts meaningful tokens

### 2. Embedding Generation

* Converts text into vector representations
* Captures semantic meaning beyond keywords

### 3. Similarity Matching

* Uses cosine similarity to compare resume vs job
* Scores based on contextual relevance

### 4. ATS Scoring

* Combines:

  * Keyword match
  * Semantic similarity
  * Skill presence

### 5. Skill Gap Detection

* Identifies missing skills required for the job
* Highlights improvement areas

---

## ⚙️ Design Decisions

* Used **semantic embeddings** instead of keyword-only matching
* Modular pipeline → easy to extend and scale
* Separated scoring logic from processing layer
* Designed system to simulate **real ATS workflows**

---

## 🚧 Challenges

* Avoiding false positives in semantic similarity
* Balancing keyword vs contextual scoring
* Handling diverse resume formats
* Designing a pipeline that remains modular

---

## 📊 Example Output

```
Match Score: 78%

Matched Skills:
- Python
- Machine Learning
- APIs

Missing Skills:
- Docker
- System Design

Recommendation:
Improve backend scalability knowledge and containerization skills
```

---

## 🛠️ Tech Stack

* Python
* NumPy
* NLP / Embeddings

---

## ▶️ Running the Project

```bash
git clone https://github.com/amith-m-s/Deep-Resume-Analyzer
cd Deep-Resume-Analyzer
pip install -r requirements.txt
python main.py
```

---

## 📌 Future Improvements

* Add LLM-based evaluation
* Improve skill extraction accuracy
* Build web interface for real-time analysis
* Add recruiter dashboard

---

## 🧠 Key Takeaway

This project focuses on **understanding resumes semantically**, not just matching keywords — making it closer to real-world hiring systems.

---

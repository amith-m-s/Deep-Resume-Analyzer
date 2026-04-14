# 🧠 Deep Resume Analyzer

A **semantic resume-job matching system** that combines embeddings, ATS-style scoring, and skill gap analysis.

Designed to simulate **real-world hiring pipelines** with structured evaluation and scalable architecture.

---

## 🚀 Overview

Traditional ATS systems rely heavily on keyword matching, often missing context.

This system improves evaluation by combining:

* Semantic similarity (context-aware)
* Keyword matching (precision)
* Skill-based validation (relevance)

---

## ⚙️ Core Features

* Semantic similarity using embeddings
* Hybrid ATS scoring system
* Skill extraction and gap detection
* Modular evaluation pipeline
* Structured scoring output

---

## 🏗️ System Architecture

```text
Input Layer:
  Resume + Job Description

        │
        ▼

Preprocessing Layer:
  - Text cleaning
  - Tokenization
  - Normalization

        │
        ▼

Embedding Layer:
  - Convert text → vector representation
  - Capture semantic meaning

        │
        ▼

Scoring Layer:
  - Semantic similarity (cosine)
  - Keyword match score
  - Skill match score

        │
        ▼

Aggregation Layer:
  - Weighted scoring system
  - Final ATS-style score

        │
        ▼

Analysis Layer:
  - Skill gap detection
  - Recommendations
```

---

## 🧠 Scoring Model

The final score is computed using a **hybrid weighted approach**:

```python
final_score = (
    0.6 * semantic_similarity +
    0.25 * keyword_score +
    0.15 * skill_score
)
```

### Why Hybrid Scoring?

* Semantic similarity → captures context
* Keyword matching → ensures precision
* Skill matching → validates real relevance

This reduces false positives and improves reliability.

---

## 📊 Evaluation Strategy

To validate system performance, controlled test cases were used:

### Test Scenarios

| Scenario           | Expected Result |
| ------------------ | --------------- |
| Matching resume    | High score      |
| Partially matching | Medium score    |
| Unrelated resume   | Low score       |

---

### Metrics Used

* **Score Consistency** → similar resumes produce similar scores
* **Precision Proxy** → relevant resumes ranked higher
* **False Positive Reduction** → irrelevant resumes penalized

---

## 🚧 Challenges & Solutions

### 1. False Positives in Semantic Similarity

* Problem: embeddings can overestimate similarity
* Solution: combined with keyword + skill scoring

---

### 2. Balancing Keyword vs Context

* Problem: keyword-only systems lack depth
* Solution: normalized scores + weighted hybrid model

---

### 3. Resume Format Variability

* Problem: inconsistent structure and noise
* Solution: preprocessing pipeline for normalization

---

### 4. Pipeline Scalability

* Problem: monolithic design limits extensibility
* Solution: modular architecture separating each stage

---

### 5. Scoring Reliability

* Problem: inconsistent evaluation results
* Solution: tested with controlled scenarios and tuning

---

## 🧠 Engineering Decisions

* Used cosine similarity for semantic comparison
* Designed modular pipeline for scalability
* Avoided black-box scoring → interpretable outputs
* Prioritized extensibility for future ML integration

---

## 📊 Example Output

```text
Match Score: 78%

Matched Skills:
- Python
- Machine Learning
- APIs

Missing Skills:
- Docker
- System Design

Recommendation:
Improve backend scalability and containerization knowledge
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

## 🔮 Future Improvements

* Integrate transformer-based embeddings (BERT / LLMs)
* Add real dataset benchmarking
* Build recruiter dashboard UI
* Introduce ranking system for multiple candidates

---

## 🧠 Key Insight

This project demonstrates how combining **semantic understanding with structured scoring** leads to more reliable and realistic hiring evaluations than traditional ATS systems.

---

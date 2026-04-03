# Deep Resume Analyzer

A deep learning-based resume analyzer that compares a resume with a job description using semantic embeddings, keyword matching, ATS scoring, skill-gap detection, and role prediction.

## Features
- Upload resume in PDF format
- Extract resume text from PDF
- Detect skills from resume
- Extract skills from job description
- Calculate ATS score
- Calculate keyword match score
- Calculate semantic similarity using deep learning
- Suggest missing skills
- Predict suitable job role
- Responsive and modern UI

## Tech Stack
- Frontend: React
- Backend: Node.js, Express
- File Upload: Multer
- PDF Parsing: pdf-parse
- Deep Learning: Python, Sentence Transformers
- Similarity: Cosine Similarity
- Deployment: Docker, Render, Vercel

## How It Works
1. User uploads a resume PDF.
2. User pastes a job description.
3. Backend extracts text from the resume.
4. Skills are extracted from both resume and job description.
5. Python model generates semantic embeddings.
6. Keyword match, ATS score, semantic score, and overall score are calculated.
7. The app suggests missing skills and predicts a suitable role.

## Installation

### Front-end
```bash
npm install
npm start
```

### Back-end
```bash
npm install
node server.js
```

## Deployment

### Front-end
```bash
https://deep-resume-analyzer.vercel.app/
```

### Back-end
```bash
https://deep-resume-analyzer.onrender.com/
```


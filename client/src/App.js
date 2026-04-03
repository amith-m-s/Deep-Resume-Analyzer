import { useMemo, useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [jobDescription, setJobDescription] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const apiBase = useMemo(() => {
    return process.env.REACT_APP_API_URL || "http://localhost:5000";
  }, []);

  const handleAnalyze = async () => {
    setError("");
    setResult(null);

    if (!file) {
      setError("Please select a PDF resume.");
      return;
    }

    if (!jobDescription.trim()) {
      setError("Please enter a job description.");
      return;
    }

    const formData = new FormData();
    formData.append("resume", file);
    formData.append("jobDescription", jobDescription);

    try {
      setLoading(true);
      const response = await axios.post(`${apiBase}/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.message || "Upload failed.");
    } finally {
      setLoading(false);
    }
  };

  const ProgressBar = ({ label, value }) => (
    <div className="score-block">
      <div className="score-row">
        <span>{label}</span>
        <span>{value}%</span>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${value}%` }} />
      </div>
    </div>
  );

  return (
    <div className="page">
      <div className="hero-card">
        <div className="header">
          <p className="eyebrow">Deep Learning Resume Intelligence</p>
          <h1>Deep Resume Analyzer</h1>
          <p className="subtitle">
            Upload a resume and compare it with any job description using semantic embeddings,
            keyword matching, ATS score, and role prediction.
          </p>
        </div>

        <div className="form-grid">
          <div className="panel">
            <label className="label">Job Description</label>
            <textarea
              className="textarea"
              placeholder="Paste the job description here..."
              rows={10}
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
            />
          </div>

          <div className="panel">
            <label className="label">Resume PDF</label>
            <input
              type="file"
              accept=".pdf"
              onChange={(e) => setFile(e.target.files[0])}
              className="file-input"
            />

            <button className="primary-btn" onClick={handleAnalyze} disabled={loading}>
              {loading ? "Analyzing..." : "Analyze Resume"}
            </button>

            {error && <p className="error">{error}</p>}

            <div className="note">
              <strong>Model:</strong> all-MiniLM-L6-v2 embeddings + cosine similarity
            </div>
          </div>
        </div>

        {result && (
          <div className="results">
            <div className="role-box">
              <div>
                <p className="role-label">Predicted Role from Resume</p>
                <h2>{result.rolePrediction?.predictedFromResume || "General Software Role"}</h2>
              </div>
              <div>
                <p className="role-label">Predicted Role from Job Description</p>
                <h2>{result.rolePrediction?.predictedFromJob || "General Software Role"}</h2>
              </div>
            </div>

            {result.jobInsights?.warning && (
              <div className="warning-box">
                {result.jobInsights.warning}
              </div>
            )}

            <div className="results-top">
              <div className="metric-card">
                <span className="metric-label">Overall Score</span>
                <span className="metric-value">{result.overallScore}%</span>
              </div>
              <div className="metric-card">
                <span className="metric-label">Semantic Match</span>
                <span className="metric-value">
                  {result.deepLearning?.semanticScore ?? 0}%
                </span>
              </div>
              <div className="metric-card">
                <span className="metric-label">ATS Score</span>
                <span className="metric-value">{result.atsScore}%</span>
              </div>
            </div>

            <ProgressBar
              label="Keyword Match"
              value={result.keywordMatchScore || 0}
            />

            <div className="section">
              <h2>Skills Found</h2>
              <div className="chip-row">
                {(result.skills || []).map((skill) => (
                  <span key={skill} className="chip">
                    {skill}
                  </span>
                ))}
              </div>
            </div>

            <div className="section">
              <h2>Job Skills Detected</h2>
              <div className="chip-row">
                {(result.jobSkills || []).length ? (
                  result.jobSkills.map((skill) => (
                    <span key={skill} className="chip chip-jd">
                      {skill}
                    </span>
                  ))
                ) : (
                  <span className="empty-text">
                    No technical skills detected in this job description.
                  </span>
                )}
              </div>
            </div>

            <div className="section">
              <h2>Missing Skills / Suggestions</h2>
              <div className="chip-row">
                {(result.suggestions || []).map((skill) => (
                  <span key={skill} className="chip chip-muted">
                    {skill}
                  </span>
                ))}
              </div>
            </div>

            <div className="section insight-box">
              <h2>Job Description Insight</h2>
              <div className="insight-grid">
                <div className="insight-item">
                  <span>Technical Skills Detected</span>
                  <strong>{result.jobInsights?.detectedSkillCount ?? 0}</strong>
                </div>
                <div className="insight-item">
                  <span>Type</span>
                  <strong>
                    {result.jobInsights?.isGeneric ? "Generic / Non-technical" : "Technical"}
                  </strong>
                </div>
              </div>
            </div>

            <div className="grid-two">
              <div className="info-box">
                <h3>Job Match</h3>
                <p>
                  <strong>Normalized Keyword Match:</strong>{" "}
                  {result.keywordMatchScore || 0}%
                </p>
                <p>
                  <strong>Raw Coverage:</strong>{" "}
                  {result.keywordCoverageScore || 0}%
                </p>
                <p>
                  <strong>Matched Skills:</strong>{" "}
                  {(result.jobMatch?.matchedSkills || []).join(", ") || "None"}
                </p>
                <p>
                  <strong>Jaccard Score:</strong> {result.jobMatch?.jaccardScore ?? 0}%
                </p>
              </div>

              <div className="info-box">
                <h3>Top Matched Resume Lines</h3>
                <ul className="line-list">
                  {(result.deepLearning?.topMatchedLines || []).map((line, index) => (
                    <li key={index}>{line}</li>
                  ))}
                </ul>
              </div>
            </div>

            <details className="details">
              <summary>View extracted resume text</summary>
              <pre className="pre">{result.text}</pre>
            </details>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const cors = require("cors");
const pdfParse = require("pdf-parse");
const { spawn } = require("child_process");

const app = express();

app.use(cors());
app.use(express.json({ limit: "2mb" }));

const uploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

const upload = multer({ dest: uploadDir });

const SKILL_RULES = [
  { name: "Machine Learning", patterns: [/\bmachine learning\b/i] },
  { name: "Deep Learning", patterns: [/\bdeep learning\b/i] },
  { name: "NLP", patterns: [/\bnlp\b/i, /\bnatural language processing\b/i] },
  { name: "REST API", patterns: [/\brest api(s)?\b/i, /\brestful api(s)?\b/i] },

  { name: "C++", patterns: [/\bc\+\+\b/i] },
  { name: "C#", patterns: [/\bc#\b/i] },
  { name: "Node.js", patterns: [/\bnode\.?js\b/i] },
  { name: "TypeScript", patterns: [/\btypescript\b/i] },
  { name: "JavaScript", patterns: [/\bjavascript\b/i] },
  { name: "Python", patterns: [/\bpython\b/i] },
  { name: "Java", patterns: [/\bjava\b/i] },
  { name: "React", patterns: [/\breact\b/i] },
  { name: "Express", patterns: [/\bexpress(?:\.js)?\b/i] },

  { name: "HTML", patterns: [/\bhtml5?\b/i] },
  { name: "CSS", patterns: [/\bcss3?\b/i] },

  { name: "MongoDB", patterns: [/\bmongodb\b/i] },
  { name: "MySQL", patterns: [/\bmysql\b/i] },
  { name: "PostgreSQL", patterns: [/\bpostgresql\b/i] },
  { name: "SQL", patterns: [/\bsql\b/i] },

  { name: "GitHub", patterns: [/\bgithub\b/i] },
  { name: "Git", patterns: [/\bgit\b/i] },

  { name: "Docker", patterns: [/\bdocker\b/i] },
  { name: "AWS", patterns: [/\baws\b/i, /\bamazon web services\b/i] },

  { name: "Flask", patterns: [/\bflask\b/i] },
  { name: "Django", patterns: [/\bdjango\b/i] },
  { name: "FastAPI", patterns: [/\bfastapi\b/i] },

  { name: "Pandas", patterns: [/\bpandas\b/i] },
  { name: "NumPy", patterns: [/\bnumpy\b/i] },
  { name: "Linux", patterns: [/\blinux\b/i] },

  { name: "OOP", patterns: [/\boop\b/i, /\bobject[- ]oriented programming\b/i] },
  { name: "DSA", patterns: [/\bdsa\b/i, /\bdata structures and algorithms\b/i, /\bdata structures & algorithms\b/i] },

  { name: "C", patterns: [/\bc\b(?!\s*[\+#])/i] }
];

function normalizeText(text) {
  return String(text || "").replace(/\s+/g, " ").trim();
}

function extractSkills(text) {
  const cleaned = normalizeText(text);
  const found = [];

  for (const rule of SKILL_RULES) {
    if (rule.patterns.some((pattern) => pattern.test(cleaned))) {
      found.push(rule.name);
    }
  }

  return [...new Set(found)];
}

function calculateATSScore(resumeSkills) {
  return Math.min(100, Math.round((resumeSkills.length / 15) * 100));
}

function suggestImprovements(resumeSkills) {
  return SKILL_RULES.map((r) => r.name).filter((skill) => !resumeSkills.includes(skill));
}

function matchJob(resumeSkills, jobSkills) {
  const normalizedJobSkills = [...new Set(jobSkills)];

  if (!normalizedJobSkills.length) {
    return {
      matchedSkills: [],
      rawCoverageScore: 0,
      normalizedScore: 0,
      jaccardScore: 0
    };
  }

  const matchedSkills = normalizedJobSkills.filter((skill) => resumeSkills.includes(skill));
  const rawCoverage = matchedSkills.length / normalizedJobSkills.length;
  const denominator = Math.max(normalizedJobSkills.length, 2);
  const normalizedScore = Math.round((matchedSkills.length / denominator) * 100);
  const unionSize = new Set([...resumeSkills, ...normalizedJobSkills]).size || 1;
  const jaccardScore = Math.round((matchedSkills.length / unionSize) * 100);

  return {
    matchedSkills,
    rawCoverageScore: Math.round(rawCoverage * 100),
    normalizedScore,
    jaccardScore
  };
}

function predictRole(skills) {
  if (skills.includes("Machine Learning")) return "Machine Learning Engineer";
  if (skills.includes("React") && skills.includes("Node.js")) return "Full Stack Developer";
  if (skills.includes("Node.js")) return "Backend Developer";
  if (skills.includes("React")) return "Frontend Developer";
  if (skills.includes("Java") || skills.includes("C++")) return "Software Engineer";
  return "General Software Role";
}

function runDeepLearningAnalysis(resumeText, jobDescription) {
  const fallback = {
    semanticScore: 0,
    topMatchedLines: [],
    model: "sentence-transformers/all-MiniLM-L6-v2",
    available: false,
    warning: "Semantic engine unavailable in this deployment. Keyword and ATS scoring still work."
  };

  const commands = [...new Set([
    process.env.PYTHON_CMD,
    process.platform === "win32" ? "python" : "python3",
    process.platform === "win32" ? "python3" : "python"
  ].filter(Boolean))];

  return new Promise((resolve) => {
    const tryCommand = (index) => {
      if (index >= commands.length) {
        return resolve(fallback);
      }

      const cmd = commands[index];
      const py = spawn(cmd, ["ml/infer.py"], {
        cwd: __dirname,
        windowsHide: true
      });

      let stdout = "";
      let settled = false;

      const next = () => {
        if (settled) return;
        settled = true;
        tryCommand(index + 1);
      };

      const timeout = setTimeout(() => {
        try {
          py.kill();
        } catch {}
        next();
      }, 45000);

      py.stdout.on("data", (data) => {
        stdout += data.toString();
      });

      py.on("error", () => {
        clearTimeout(timeout);
        next();
      });

      py.on("close", (code) => {
        clearTimeout(timeout);
        if (settled) return;

        if (code !== 0) {
          return next();
        }

        try {
          const parsed = JSON.parse(stdout.trim() || "{}");
          resolve({
            semanticScore: Number(parsed.semanticScore) || 0,
            topMatchedLines: Array.isArray(parsed.topMatchedLines) ? parsed.topMatchedLines : [],
            model: parsed.model || "sentence-transformers/all-MiniLM-L6-v2",
            available: true,
            rolePrediction: parsed.rolePrediction || undefined
          });
        } catch {
          next();
        }
      });

      try {
        py.stdin.write(JSON.stringify({
          resume_text: resumeText,
          job_description: jobDescription
        }));
        py.stdin.end();
      } catch {
        clearTimeout(timeout);
        next();
      }
    };

    tryCommand(0);
  });
}

app.get("/", (req, res) => {
  res.json({ message: "Deep Resume Analyzer backend is running" });
});

app.get("/health", (req, res) => {
  res.json({ ok: true });
});

app.post("/upload", upload.single("resume"), async (req, res) => {
  let filePath = null;

  try {
    if (!req.file) {
      return res.status(400).json({ message: "No file uploaded" });
    }

    const jobDescription = normalizeText(req.body.jobDescription);
    if (!jobDescription) {
      return res.status(400).json({ message: "Job description is required" });
    }

    filePath = req.file.path;

    const parsed = await pdfParse(fs.readFileSync(filePath));
    const resumeText = normalizeText(parsed.text);

    const resumeSkills = extractSkills(resumeText);
    const jobSkills = extractSkills(jobDescription);

    const atsScore = calculateATSScore(resumeSkills);
    const suggestions = suggestImprovements(resumeSkills);
    const jobMatch = matchJob(resumeSkills, jobSkills);
    const semantic = await runDeepLearningAnalysis(resumeText, jobDescription);

    const semanticScore = semantic.semanticScore || 0;
    const keywordMatchScore = jobMatch.normalizedScore;
    const overallScore = Math.round(
      semanticScore * 0.5 +
      keywordMatchScore * 0.3 +
      atsScore * 0.2
    );

    const rolePredictionResume = predictRole(resumeSkills);
    const rolePredictionJob = predictRole(jobSkills);

    const jobInsights = {
      detectedSkillCount: jobSkills.length,
      isGeneric: jobSkills.length < 2,
      warning:
        jobSkills.length < 2
          ? "This job description appears generic or non-technical. Add more technical keywords for a better keyword match."
          : ""
    };

    res.json({
      message: "Resume processed",
      atsScore,
      keywordMatchScore,
      keywordCoverageScore: jobMatch.rawCoverageScore,
      overallScore,
      skills: resumeSkills,
      jobSkills,
      suggestions,
      jobMatch,
      rolePrediction: {
        predictedFromResume: rolePredictionResume,
        predictedFromJob: rolePredictionJob
      },
      jobInsights,
      deepLearning: {
        semanticScore,
        topMatchedLines: semantic.topMatchedLines || [],
        model: semantic.model || "sentence-transformers/all-MiniLM-L6-v2",
        available: semantic.available !== false,
        warning: semantic.warning || ""
      },
      stats: {
        resumeWords: resumeText ? resumeText.split(/\s+/).length : 0,
        jobDescriptionWords: jobDescription.split(/\s+/).length
      },
      text: parsed.text
    });
  } catch (error) {
    console.error("ERROR:", error);
    res.status(500).json({
      message: "Error processing file",
      error: error.message
    });
  } finally {
    if (filePath && fs.existsSync(filePath)) {
      try {
        fs.unlinkSync(filePath);
      } catch {}
    }
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server started on port ${PORT}`));
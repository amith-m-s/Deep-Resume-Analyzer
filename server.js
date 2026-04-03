const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const cors = require("cors");
const pdfParse = require("pdf-parse");
const { spawn } = require("child_process");

const app = express();
app.use(cors());
app.use(express.json());

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

  { name: "C", patterns: [/(^|[^a-zA-Z0-9])c([^a-zA-Z0-9]|$)/i] }
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
  const totalSkills = SKILL_RULES.length || 1;
  return Math.round((resumeSkills.length / totalSkills) * 100);
}

function suggestImprovements(resumeSkills) {
  const unique = new Set(resumeSkills);
  return SKILL_RULES.map((r) => r.name).filter((skill) => !unique.has(skill));
}

function extractJobSkills(jobDescription) {
  return extractSkills(jobDescription);
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

  const richnessFactor = Math.min(1, normalizedJobSkills.length / 6);
  const normalizedScore = Math.round(rawCoverage * richnessFactor * 100);

  const unionSize = new Set([...resumeSkills, ...normalizedJobSkills]).size || 1;
  const jaccardScore = Math.round((matchedSkills.length / unionSize) * 100);

  return {
    matchedSkills,
    rawCoverageScore: Math.round(rawCoverage * 100),
    normalizedScore,
    jaccardScore
  };
}

function predictRole(resumeSkills, jobSkills) {
  const roles = [
    {
      role: "Full Stack Developer",
      skills: ["React", "Node.js", "JavaScript", "HTML", "CSS", "Express", "SQL"]
    },
    {
      role: "Backend Developer",
      skills: ["Node.js", "Express", "SQL", "MySQL", "PostgreSQL", "REST API"]
    },
    {
      role: "Frontend Developer",
      skills: ["React", "JavaScript", "TypeScript", "HTML", "CSS"]
    },
    {
      role: "Machine Learning Engineer",
      skills: ["Python", "Machine Learning", "Deep Learning", "NLP", "NumPy", "Pandas"]
    },
    {
      role: "Data Scientist",
      skills: ["Python", "Pandas", "NumPy", "Machine Learning", "SQL"]
    },
    {
      role: "Software Engineer",
      skills: ["Java", "C++", "OOP", "DSA", "Git", "SQL"]
    }
  ];

  let bestResumeRole = "General Software Role";
  let bestResumeScore = -1;

  for (const item of roles) {
    const score = item.skills.filter((s) => resumeSkills.includes(s)).length;
    if (score > bestResumeScore) {
      bestResumeScore = score;
      bestResumeRole = item.role;
    }
  }

  let bestJobRole = "General Software Role";
  let bestJobScore = -1;

  for (const item of roles) {
    const score = item.skills.filter((s) => jobSkills.includes(s)).length;
    if (score > bestJobScore) {
      bestJobScore = score;
      bestJobRole = item.role;
    }
  }

  return {
    predictedFromResume: bestResumeRole,
    predictedFromJob: bestJobScore > 0 ? bestJobRole : "General Software Role"
  };
}

function runDeepLearningAnalysis(resumeText, jobDescription) {
  return new Promise((resolve, reject) => {
    const pythonCmd = process.env.PYTHON_CMD || (process.platform === "win32" ? "python" : "python3");
    const py = spawn(pythonCmd, ["ml/infer.py"], {
      cwd: __dirname,
      windowsHide: true
    });

    let stdout = "";
    let stderr = "";

    py.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    py.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    py.on("error", (err) => reject(err));

    py.on("close", (code) => {
      if (code !== 0) {
        return reject(new Error(stderr || `Python exited with code ${code}`));
      }

      try {
        resolve(JSON.parse(stdout.trim() || "{}"));
      } catch (err) {
        reject(new Error(`Failed to parse Python output: ${err.message}`));
      }
    });

    py.stdin.write(
      JSON.stringify({
        resume_text: resumeText,
        job_description: jobDescription
      })
    );
    py.stdin.end();
  });
}

app.get("/", (req, res) => {
  res.json({ message: "Deep Resume Analyzer backend is running" });
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
    const jobSkills = extractJobSkills(jobDescription);

    const atsScore = calculateATSScore(resumeSkills);
    const suggestions = suggestImprovements(resumeSkills);
    const jobMatch = matchJob(resumeSkills, jobSkills);
    const rolePrediction = predictRole(resumeSkills, jobSkills);

    const deepLearning = await runDeepLearningAnalysis(resumeText, jobDescription);

    const keywordMatchScore = jobMatch.normalizedScore;
    const semanticScore = deepLearning.semanticScore || 0;

    const overallScore = Math.round(
      semanticScore * 0.5 +
      keywordMatchScore * 0.3 +
      atsScore * 0.2
    );

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
      rolePrediction,
      jobInsights,
      deepLearning: {
        semanticScore,
        topMatchedLines: deepLearning.topMatchedLines || [],
        model: deepLearning.model || "sentence-transformers/all-MiniLM-L6-v2"
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
      } catch (_) {}
    }
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server started on port ${PORT}`));
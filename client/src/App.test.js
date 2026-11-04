import { fireEvent, render, screen } from "@testing-library/react";
import axios from "axios";
import App from "./App";

jest.mock("axios", () => ({
  post: jest.fn()
}));

beforeEach(() => {
  jest.clearAllMocks();
});

test("renders accessible labels for the job description and resume inputs", () => {
  render(<App />);

  expect(screen.getByRole("heading", { name: "Deep Resume Analyzer" })).toBeInTheDocument();
  expect(screen.getByLabelText("Job Description")).toBeInTheDocument();
  expect(screen.getByLabelText("Resume PDF")).toBeInTheDocument();
});

test("rejects files that are not PDFs before making an API request", () => {
  render(<App />);
  const input = screen.getByLabelText("Resume PDF");
  const invalidFile = new File(["not a PDF"], "resume.txt", { type: "text/plain" });

  fireEvent.change(input, { target: { files: [invalidFile] } });

  expect(screen.getByRole("alert")).toHaveTextContent("Please select a PDF file.");
  expect(axios.post).not.toHaveBeenCalled();
});

test("rejects PDF files larger than the server upload limit", () => {
  render(<App />);
  const input = screen.getByLabelText("Resume PDF");
  const oversizedFile = new File(
    [new Uint8Array(5 * 1024 * 1024 + 1)],
    "resume.pdf",
    { type: "application/pdf" }
  );

  fireEvent.change(input, { target: { files: [oversizedFile] } });

  expect(screen.getByRole("alert")).toHaveTextContent("Resume must be 5 MB or smaller.");
  expect(axios.post).not.toHaveBeenCalled();
});

test("requires a job description before submitting a valid PDF", () => {
  render(<App />);
  const input = screen.getByLabelText("Resume PDF");
  const pdfFile = new File(["%PDF-1.7"], "resume.pdf", { type: "application/pdf" });

  fireEvent.change(input, { target: { files: [pdfFile] } });
  fireEvent.click(screen.getByRole("button", { name: "Analyze Resume" }));

  expect(screen.getByRole("alert")).toHaveTextContent("Please enter a job description.");
  expect(axios.post).not.toHaveBeenCalled();
});

test("submits a valid PDF and job description to the analyzer API", async () => {
  axios.post.mockResolvedValue({
    data: {
      overallScore: 80,
      atsScore: 80,
      keywordMatchScore: 75,
      keywordCoverageScore: 75,
      jobInsights: {},
      rolePrediction: {},
      deepLearning: {},
      skills: [],
      jobSkills: [],
      suggestions: [],
      jobMatch: {}
    }
  });

  render(<App />);
  const input = screen.getByLabelText("Resume PDF");
  const pdfFile = new File(["%PDF-1.7"], "resume.pdf", { type: "application/pdf" });

  fireEvent.change(input, { target: { files: [pdfFile] } });
  fireEvent.change(screen.getByLabelText("Job Description"), {
    target: { value: "Looking for a Python backend developer with SQL experience." }
  });
  fireEvent.click(screen.getByRole("button", { name: "Analyze Resume" }));

  expect(await screen.findByText("Strong Match")).toBeInTheDocument();
  expect(axios.post).toHaveBeenCalledTimes(1);
  expect(axios.post.mock.calls[0][0]).toMatch(/\/upload$/);
  expect(axios.post.mock.calls[0][1]).toBeInstanceOf(FormData);
});

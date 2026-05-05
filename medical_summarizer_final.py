"""
Medical Report Summarizer — Improved Version
"""

import os
import re
import csv
import logging
import json
import yaml
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

try:
    from bert_score import score as bert_score_fn
except ImportError:
    bert_score_fn = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

# -------------------------------
# LOGGING
# -------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("summarizer.log"),
    ],
)
log = logging.getLogger(__name__)

# -------------------------------
# SETTINGS
# -------------------------------

def load_config(path: str = "config.yaml") -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

_config = load_config()

INPUT_FOLDER = _config.get("input_folder", "reports")
OUTPUT_FOLDER = _config.get("output_folder", "summaries")
RESULT_FILE = _config.get("result_file", "results.csv")
GRAPH_FILE = _config.get("graph_file", "rouge_graph.png")

_model_id = _config.get("model_id", "google/flan-t5-base")
MODELS: dict[str, str] = {
    _model_id.split("/")[-1]: _model_id
}

MAX_INPUT_TOKENS = _config.get("max_input_tokens", 512)
MAX_NEW_TOKENS = _config.get("max_new_tokens", 140)
HALLUCINATION_FILTER_THRESHOLD = _config.get("hallucination_filter_threshold", 3)

SECTION_PATTERNS: dict[str, str] = {
    "symptoms":  "Chief Complaint",
    "diagnosis": "Assessment",
    "tests":     "Investigations",
    "treatment": "Plan / Treatment",
}

# -------------------------------
# DATA CLASSES
# -------------------------------

@dataclass
class Report:
    filename: str
    text: str


@dataclass
class SummaryResult:
    model: str
    report: str
    summary: str
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    bertscore_precision: float = 0.0
    bertscore_recall: float = 0.0
    bertscore_f1: float = 0.0
    error: Optional[str] = None

    def as_csv_row(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "summary"}


# -------------------------------
# FILE I/O
# -------------------------------

def load_reports(folder: str) -> list[Report]:
    if not os.path.isdir(folder):
        log.error("Reports folder not found: %s", folder)
        return []

    reports = []
    for file in sorted(os.listdir(folder)):
        if file.endswith(".txt"):
            path = os.path.join(folder, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    reports.append(Report(filename=file, text=f.read()))
                log.info("Loaded: %s", file)
            except OSError as e:
                log.warning("Could not read %s: %s", file, e)

    log.info("Loaded %d report(s)", len(reports))
    return reports


def save_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# -------------------------------
# SECTION EXTRACTION
# -------------------------------

def extract_section(text: str, section: str) -> str:
    pattern = rf"{re.escape(section)}:(.*?)(?=\n[A-Z][a-zA-Z /]+:|$)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else "Not available"


# -------------------------------
# HALLUCINATION FILTER
# -------------------------------

class NLIFilter:
    def __init__(self):
        self.model = None
        if CrossEncoder is not None:
            try:
                self.model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
                log.info("NLI model loaded successfully.")
            except Exception as e:
                log.warning("Failed to load NLI model: %s. Falling back to word-overlap.", e)

    def filter(self, summary: str, original: str, min_overlap: int) -> str:
        if self.model is None:
            return filter_hallucinations_fallback(summary, original, min_overlap)
        
        valid_sentences = []
        sentences = [s.strip() for s in summary.split(".") if s.strip()]
        for s in sentences:
            try:
                preds = self.model.predict([(original, s)])
                label_id = np.argmax(preds[0]) if preds.ndim > 1 else np.argmax(preds)
                # For cross-encoder/nli-deberta-v3-small, 0 usually refers to contradiction
                if label_id != 0:
                    valid_sentences.append(s)
            except Exception as e:
                log.warning("NLI prediction failed: %s. Falling back.", e)
                return filter_hallucinations_fallback(summary, original, min_overlap)
                
        filtered_summary = ". ".join(valid_sentences).strip()
        return (filtered_summary + ".") if filtered_summary else summary

_nli_filter = None

def filter_hallucinations_fallback(summary: str, original: str, min_overlap: int = 3) -> str:
    """
    Drop sentences whose words have fewer than `min_overlap` matches
    with the source document. Falls back to full summary if all are filtered.
    """
    original_words = set(original.lower().split())
    valid = [
        s for s in summary.split(".")
        if len(set(s.lower().split()) & original_words) >= min_overlap
    ]
    return ". ".join(valid).strip() or summary

def filter_hallucinations(summary: str, original: str, min_overlap: int = None) -> str:
    """
    NLI-based hallucination filter with word-overlap fallback.
    """
    if min_overlap is None:
        min_overlap = HALLUCINATION_FILTER_THRESHOLD
        
    global _nli_filter
    if _nli_filter is None:
        _nli_filter = NLIFilter()
        
    if _nli_filter.model is not None:
        return _nli_filter.filter(summary, original, min_overlap)
    return filter_hallucinations_fallback(summary, original, min_overlap)


# -------------------------------
# MODEL MANAGEMENT
# -------------------------------

class Summarizer:
    """Loads a model once and reuses it across all reports."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading model '%s' on %s ...", model_id, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
        self.model.eval()
        log.info("Model ready.")

    def summarize(self, text: str) -> str:
        prompt = (
            "You are a helpful medical AI. Read the following medical report and extract "
            "a concise summary of the key symptoms, diagnosis, tests, and treatment plan.\n\n"
            f"Report: {text}\n\nSummary:"
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=5,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                early_stopping=True,
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)


# -------------------------------
# EVALUATION
# -------------------------------

_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def evaluate_rouge(reference: str, generated: str) -> tuple[float, float, float]:
    """
    NOTE: Pass a meaningful reference summary when available.
    Evaluating against a raw report slice inflates scores artificially.
    """
    scores = _rouge.score(reference, generated)
    return (
        scores["rouge1"].fmeasure,
        scores["rouge2"].fmeasure,
        scores["rougeL"].fmeasure,
    )

def evaluate_bertscore(reference: str, generated: str) -> tuple[float, float, float]:
    if bert_score_fn is None:
        log.warning("BERTScore not installed.")
        return 0.0, 0.0, 0.0
    try:
        P, R, F1 = bert_score_fn([generated], [reference], lang="en", verbose=False)
        return P.item(), R.item(), F1.item()
    except Exception as e:
        log.warning("BERTScore evaluation failed: %s", e)
        return 0.0, 0.0, 0.0


# -------------------------------
# STRUCTURED OUTPUT
# -------------------------------

def build_structured_summary(report: Report, summary: str) -> str:
    sections = {
        label: extract_section(report.text, pattern)
        for label, pattern in SECTION_PATTERNS.items()
    }
    lines = ["=" * 60, "STRUCTURED MEDICAL SUMMARY", f"Report: {report.filename}", "=" * 60]
    for label, content in sections.items():
        lines += [f"\n{label.upper()}", content]
    lines += ["\nAI SUMMARY", summary, "=" * 60]
    return "\n".join(lines)


# -------------------------------
# GRAPH GENERATION
# -------------------------------

def generate_graph(results: list[SummaryResult], output_path: str = GRAPH_FILE) -> None:
    """Grouped bar chart for ROUGE-1/2/L across all model×report pairs."""
    if not results:
        log.warning("No results to graph.")
        return

    labels = [f"{r.model}\n{r.report}" for r in results]
    r1 = [r.rouge1 for r in results]
    r2 = [r.rouge2 for r in results]
    rL = [r.rougeL for r in results]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    ax.bar(x - width, r1, width, label="ROUGE-1", color="#4C72B0")
    ax.bar(x,         r2, width, label="ROUGE-2", color="#DD8452")
    ax.bar(x + width, rL, width, label="ROUGE-L", color="#55A868")

    ax.set_title("ROUGE Score Comparison by Model & Report")
    ax.set_ylabel("F-measure")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    log.info("Graph saved: %s", output_path)


# -------------------------------
# CSV EXPORT
# -------------------------------

def save_results_csv(results: list[SummaryResult], path: str = RESULT_FILE) -> None:
    if not results:
        return
    fieldnames = list(results[0].as_csv_row().keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.as_csv_row())
    log.info("Results saved: %s", path)


# -------------------------------
# MAIN
# -------------------------------

def main() -> None:
    log.info("=== Medical Report Summarizer ===")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    reports = load_reports(INPUT_FOLDER)
    if not reports:
        log.error("No reports found in '%s'. Exiting.", INPUT_FOLDER)
        return

    all_results: list[SummaryResult] = []

    for model_name, model_id in MODELS.items():
        summarizer = Summarizer(model_id)  # Load ONCE per model

        for report in reports:
            log.info("[%s] Processing: %s", model_name, report.filename)
            result = SummaryResult(model=model_name, report=report.filename, summary="")

            try:
                raw_summary = summarizer.summarize(report.text)
                result.summary = filter_hallucinations(raw_summary, report.text)
            except (RuntimeError, ValueError) as e:
                log.error("Summarization failed for %s: %s", report.filename, e)
                result.summary = "Summary failed"
                result.error = str(e)

            # Save structured output
            structured = build_structured_summary(report, result.summary)
            out_path = os.path.join(
                OUTPUT_FOLDER,
                report.filename.replace(".txt", "_summary.txt"),
            )
            save_text(out_path, structured)

            # Evaluate — use first 512 chars as proxy reference (replace with
            # real reference summaries if available for meaningful scores)
            reference = report.text[:512]
            result.rouge1, result.rouge2, result.rougeL = evaluate_rouge(
                reference, result.summary
            )
            result.bertscore_precision, result.bertscore_recall, result.bertscore_f1 = evaluate_bertscore(
                reference, result.summary
            )
            log.info(
                "ROUGE → R1: %.3f  R2: %.3f  RL: %.3f | BERTScore → F1: %.3f",
                result.rouge1, result.rouge2, result.rougeL, result.bertscore_f1
            )
            
            # JSON Export
            sections = {
                label: extract_section(report.text, pattern)
                for label, pattern in SECTION_PATTERNS.items()
            }
            json_out = {
                "filename": report.filename,
                "model": model_name,
                "symptoms": sections.get("symptoms", ""),
                "diagnosis": sections.get("diagnosis", ""),
                "tests": sections.get("tests", ""),
                "treatment": sections.get("treatment", ""),
                "ai_summary": result.summary,
                "rouge1": result.rouge1,
                "rouge2": result.rouge2,
                "rougeL": result.rougeL,
                "bertscore_f1": result.bertscore_f1
            }
            json_out_path = os.path.join(
                OUTPUT_FOLDER,
                report.filename.replace(".txt", "_summary.json"),
            )
            try:
                with open(json_out_path, "w", encoding="utf-8") as f:
                    json.dump(json_out, f, indent=4)
            except Exception as e:
                log.error("Failed to save JSON for %s: %s", report.filename, e)

            all_results.append(result)

    save_results_csv(all_results)
    generate_graph(all_results)

    log.info("Done. Summaries in '%s', results in '%s'.", OUTPUT_FOLDER, RESULT_FILE)


if __name__ == "__main__":
    main()
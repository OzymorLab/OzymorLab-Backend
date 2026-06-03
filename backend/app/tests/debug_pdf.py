#!/usr/bin/env python3
"""
Answer Sheet OCR — Multi-subject, with diagrams, student info, and paired Q&A LaTeX output.

Usage:
    python3 answer_sheet_ocr.py <answer_sheet.pdf> [question_paper.pdf]

Outputs:
    <stem>_student_info.json   — extracted student details
    <stem>_transcript.tex      — full LaTeX with questions paired to answers
    <stem>_diagrams/           — saved diagram/graph images (PNG)
"""

import sys
import json
import base64
import re
from pathlib import Path

import fitz  # PyMuPDF

from google import genai
from google.genai import types


# ==================================================
# CONFIG
# ==================================================

API_KEY = ""
MODEL   = ""

client = genai.Client(api_key=API_KEY)


# ==================================================
# PDF → IMAGES
# ==================================================

def pdf_to_images(pdf_path: str, dpi_scale: float = 2.0) -> list[bytes]:
    """Render every page of a PDF to PNG bytes."""
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi_scale, dpi_scale), alpha=False)
        images.append(pix.tobytes("png"))
    return images


# ==================================================
# PROMPTS
# ==================================================

STUDENT_INFO_PROMPT = """
You are extracting student metadata from an exam answer sheet cover page or header.

Return ONLY a JSON object (no markdown, no explanation) with these keys:
{
  "name":         "<student full name or UNKNOWN>",
  "roll_number":  "<roll/enrolment number or UNKNOWN>",
  "class":        "<class/grade or UNKNOWN>",
  "section":      "<section or UNKNOWN>",
  "subject":      "<subject name or UNKNOWN>",
  "date":         "<exam date or UNKNOWN>",
  "school":       "<school/institution or UNKNOWN>",
  "max_marks":    "<maximum marks or UNKNOWN>",
  "obtained_marks": "<marks obtained or UNKNOWN>"
}

If a field is absent leave it as "UNKNOWN". Do NOT invent values.
"""

OCR_PROMPT = """
You are an expert OCR engine for exam answer sheets covering ANY subject
(Mathematics, Physics, Chemistry, Biology, History, Geography, Economics, etc.).

Your ONLY job is faithful transcription — never solve, simplify, or explain.

══════════════════════════════════════════════════
GENERAL RULES
══════════════════════════════════════════════════
1. Preserve question numbering exactly as written.
2. Preserve line breaks and paragraph structure.
3. Preserve all written content verbatim (spelling mistakes included).
4. Output clean UTF-8 text with LaTeX math where needed.

══════════════════════════════════════════════════
MATHEMATICS & SCIENCE NOTATION
══════════════════════════════════════════════════
Convert all mathematical/chemical/physical expressions to LaTeX:

• Limits            →  $$\\lim_{x\\to 0} f(x)$$
• Fractions         →  $$\\frac{a}{b}$$
• Roots             →  $$\\sqrt{x}$$,  $$\\sqrt[3]{x}$$
• Powers            →  $$x^{2}$$
• Integrals         →  $$\\int_{a}^{b} f(x)\\,dx$$
• Derivatives       →  $$\\frac{d}{dx}$$,  $$\\frac{\\partial f}{\\partial x}$$
• Trig/log          →  $$\\sin x$$,  $$\\ln x$$,  $$\\log_{10} x$$
• Vectors           →  $$\\vec{F} = m\\vec{a}$$
• Chemical eqns     →  Use \\ce{} notation: \\ce{H2O},  \\ce{CO2 + H2O -> H2CO3}
• Physics units     →  $$9.8\\,\\text{m/s}^2$$
• Matrices          →  Use pmatrix / bmatrix environments

══════════════════════════════════════════════════
DIAGRAMS, GRAPHS, FIGURES & TABLES
══════════════════════════════════════════════════
When you detect a hand-drawn diagram, graph, biological drawing, map, circuit,
flow-chart, table, or any non-text visual element:

1. Insert this EXACT placeholder (one per distinct visual):
   [DIAGRAM_START]
   TYPE: <one of: graph | biological_diagram | circuit | flowchart | map | table | other>
   DESCRIPTION: <brief factual description of what is drawn, e.g.
                 "Bell-shaped curve labelled 'Normal Distribution', x-axis 'Score',
                  y-axis 'Frequency'">
   LABELS: <comma-separated list of all text labels visible inside the figure>
   LATEX_REPRESENTATION: <if a table → full LaTeX tabular; if a simple graph →
                           TikZ skeleton or pgfplots skeleton; otherwise leave blank>
   [DIAGRAM_END]

2. Continue transcribing the surrounding text normally.

══════════════════════════════════════════════════
UNREADABLE CONTENT
══════════════════════════════════════════════════
• Single unreadable word  →  [UNCLEAR]
• Unreadable sentence     →  [UNCLEAR_SENTENCE]
• Completely blank answer →  [BLANK]

NEVER guess unreadable handwriting.

══════════════════════════════════════════════════
OUTPUT FORMAT
══════════════════════════════════════════════════
Return only the transcribed text. No commentary, no preamble.
"""

QUESTION_PAPER_PROMPT = """
You are extracting questions from a printed or handwritten question paper.

Return ONLY a JSON array (no markdown fences) in this form:
[
  {"number": "1", "text": "<full question text with LaTeX math>"},
  {"number": "2", "text": "..."},
  ...
]

Rules:
- number: the question number/label exactly as printed (e.g. "1", "1a", "Q3(ii)")
- text: verbatim question including all sub-parts; use LaTeX for math/science notation
- Use \\ce{} for chemical equations
- If a question contains a diagram/figure, add: [FIGURE: <brief description>]
- Preserve marks in brackets if shown, e.g. "[4 marks]"
"""


# ==================================================
# GEMINI HELPERS
# ==================================================

def _call(prompt: str, image_bytes: bytes | None = None, page_label: str = "") -> str:
    """Single Gemini call; returns text."""
    parts = [prompt]
    if image_bytes:
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
    try:
        resp = client.models.generate_content(model=MODEL, contents=parts)
        return resp.text or ""
    except Exception as exc:
        print(f"[ERROR] {page_label}: {exc}")
        return f"[GEMINI_FAILED_{page_label}]"


def extract_student_info(image_bytes: bytes) -> dict:
    raw = _call(STUDENT_INFO_PROMPT, image_bytes, "student_info")
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        return json.loads(raw)
    except Exception:
        print("[WARN] Could not parse student info JSON; using defaults.")
        return {k: "UNKNOWN" for k in
                ["name","roll_number","class","section","subject",
                 "date","school","max_marks","obtained_marks"]}


def ocr_page(image_bytes: bytes, page_number: int) -> str:
    return _call(OCR_PROMPT, image_bytes, f"page_{page_number}")


def extract_questions(images: list[bytes]) -> list[dict]:
    """Extract questions from all pages of the question paper."""
    all_questions: list[dict] = []
    for idx, img in enumerate(images):
        print(f"  [INFO] Extracting questions from page {idx+1}/{len(images)}")
        raw = _call(QUESTION_PAPER_PROMPT, img, f"qpaper_p{idx+1}")
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            questions = json.loads(raw)
            all_questions.extend(questions)
        except Exception:
            print(f"  [WARN] Could not parse questions on page {idx+1}")
    # Deduplicate by number (keep first occurrence)
    seen: set[str] = set()
    unique: list[dict] = []
    for q in all_questions:
        key = str(q.get("number","")).strip()
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


# ==================================================
# DIAGRAM EXTRACTION
# ==================================================

def extract_diagrams(transcript: str, out_dir: Path) -> str:
    """
    Parse [DIAGRAM_START]...[DIAGRAM_END] blocks.
    Save metadata as JSON stubs; return modified transcript with \\includegraphics hints.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(
        r'\[DIAGRAM_START\](.*?)\[DIAGRAM_END\]',
        re.DOTALL | re.IGNORECASE
    )

    counter = [0]

    def replace_block(match: re.Match) -> str:
        counter[0] += 1
        n = counter[0]
        block = match.group(1).strip()

        # Parse fields
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if ':' in line:
                key, _, val = line.partition(':')
                fields[key.strip().upper()] = val.strip()

        dtype  = fields.get("TYPE", "other")
        desc   = fields.get("DESCRIPTION", "")
        labels = fields.get("LABELS", "")
        latex  = fields.get("LATEX_REPRESENTATION", "")

        meta_file = out_dir / f"diagram_{n:03d}.json"
        meta_file.write_text(json.dumps({
            "diagram_id": n, "type": dtype,
            "description": desc, "labels": labels,
            "latex_representation": latex
        }, indent=2, ensure_ascii=False))

        # Build LaTeX replacement
        if latex and latex.lower() not in ("", "none", "blank"):
            latex_block = f"""
% --- Diagram {n}: {dtype} ---
% {desc}
{latex}
% Labels: {labels}
"""
        else:
            latex_block = f"""
% --- Diagram {n}: {dtype} ---
\\begin{{figure}}[h]
\\centering
\\fbox{{\\parbox{{0.8\\textwidth}}{{
  \\textbf{{[Hand-drawn {dtype}]}}\\\\
  \\textit{{{desc}}}\\\\
  \\textbf{{Labels:}} {labels}
}}}}
\\caption{{Diagram {n} — {dtype}}}
\\label{{fig:diagram{n}}}
\\end{{figure}}
"""
        return latex_block.strip()

    return pattern.sub(replace_block, transcript)


# ==================================================
# PARSE ANSWER BLOCKS FROM TRANSCRIPT
# ==================================================

def parse_answers(full_transcript: str) -> dict[str, str]:
    """
    Heuristic: find lines like "1.", "Q1", "1)", "(1)" etc. and group
    following text as that question's answer.
    Returns {question_number_str: answer_text}.
    """
    # Patterns: "1.", "1)", "(1)", "Q1", "Q.1", "Ans 1", etc.
    q_pattern = re.compile(
        r'^(?:Q\.?\s*|Ans\.?\s*|Answer\.?\s*)?'
        r'(\d+(?:[a-z](?:\([ivx]+\))?)?)'
        r'[.)\]:\s]',
        re.IGNORECASE | re.MULTILINE
    )

    matches = list(q_pattern.finditer(full_transcript))
    answers: dict[str, str] = {}

    for i, m in enumerate(matches):
        num = m.group(1).strip()
        start = m.end()
        end   = matches[i+1].start() if i+1 < len(matches) else len(full_transcript)
        body  = full_transcript[start:end].strip()
        # Normalise number: "01" → "1"
        try:
            num_norm = str(int(re.sub(r'[a-z].*', '', num)))
            suffix   = re.sub(r'^\d+', '', num)
            num = num_norm + suffix
        except ValueError:
            pass
        answers[num] = body

    return answers


# ==================================================
# LATEX GENERATION
# ==================================================

LATEX_TEMPLATE = r"""
\documentclass[12pt,a4paper]{{article}}

% Packages
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{geometry}}
\usepackage{{amsmath,amssymb,amsthm}}
\usepackage{{mhchem}}           % \ce{{}} for chemistry
\usepackage{{siunitx}}          % SI units
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{array,booktabs,longtable}}
\usepackage{{tikz,pgfplots}}
\usepackage{{xcolor}}
\usepackage{{fancyhdr}}
\usepackage{{hyperref}}
\usepackage{{enumitem}}

\geometry{{margin=2.5cm}}

% Header / Footer
\pagestyle{{fancy}}
\fancyhf{{}}
\rhead{{{subject}}}
\lhead{{Answer Sheet Transcript}}
\cfoot{{\thepage}}

\hypersetup{{
  colorlinks=true, linkcolor=blue, urlcolor=blue
}}

\begin{{document}}

% ======================================================
%  STUDENT INFORMATION
% ======================================================
\begin{{center}}
  {{\LARGE\bfseries Answer Sheet Transcript}}\\[6pt]
  {{\large\itshape {subject}}}
\end{{center}}

\vspace{{8pt}}
\begin{{center}}
\begin{{tabular}}{{|l|l||l|l|}}
\hline
\textbf{{Student Name}} & {name} &
\textbf{{Roll Number}}  & {roll_number} \\\\
\hline
\textbf{{Class / Section}} & {class_sec} &
\textbf{{Date}}            & {date} \\\\
\hline
\textbf{{School}} & \multicolumn{{3}}{{l|}}{{{school}}} \\\\
\hline
\textbf{{Max Marks}} & {max_marks} &
\textbf{{Obtained}} & {obtained_marks} \\\\
\hline
\end{{tabular}}
\end{{center}}

\vspace{{12pt}}
\hrule
\vspace{{12pt}}

% ======================================================
%  ANSWERS  (with paired questions)
% ======================================================
{qa_body}

\end{{document}}
""".strip()


def build_latex(student_info: dict, questions: list[dict],
                answers: dict[str, str], full_transcript: str) -> str:

    # Build question lookup
    q_lookup: dict[str, str] = {
        str(q.get("number","")).strip(): q.get("text","") for q in questions
    }

    # Collect all question numbers that appear (union of questions + answers)
    all_nums = sorted(
        set(q_lookup.keys()) | set(answers.keys()),
        key=lambda x: (int(re.sub(r'\D.*','',x)) if re.match(r'\d',x) else 9999, x)
    )

    qa_lines: list[str] = []

    if all_nums:
        for num in all_nums:
            q_text = q_lookup.get(num, "")
            a_text = answers.get(num, "")

            block = f"\\subsection*{{Question {num}}}\n"

            if q_text:
                block += (
                    "\\begin{tcolorbox_placeholder}%\n"
                    f"\\textbf{{Q{num}.}} {q_text}\n"
                    "\\end{tcolorbox_placeholder}%\n\n"
                )
                # Replace placeholder with simple framed box using minipage
                block = block.replace(
                    "\\begin{tcolorbox_placeholder}%\n",
                    "\\noindent\\fbox{\\begin{minipage}{0.97\\textwidth}\n"
                ).replace(
                    "\\end{tcolorbox_placeholder}%\n",
                    "\\end{minipage}}\n"
                )
            else:
                block += f"\\textit{{(Question {num} — not found in question paper)}}\n\n"

            block += "\n\\textbf{Answer:}\n\n"
            block += (a_text if a_text else "\\textit{[No answer detected]}")
            block += "\n\n\\bigskip\\hrule\\bigskip\n"
            qa_lines.append(block)
    else:
        # Fall back: dump entire transcript
        qa_lines.append("\\section*{Full Transcript}\n\n" + full_transcript)

    class_sec = "/".join(filter(lambda x: x != "UNKNOWN",
                                [student_info.get("class","UNKNOWN"),
                                 student_info.get("section","UNKNOWN")]))
    if not class_sec:
        class_sec = "UNKNOWN"

    return LATEX_TEMPLATE.format(
        subject        = _esc(student_info.get("subject","UNKNOWN")),
        name           = _esc(student_info.get("name","UNKNOWN")),
        roll_number    = _esc(student_info.get("roll_number","UNKNOWN")),
        class_sec      = _esc(class_sec),
        date           = _esc(student_info.get("date","UNKNOWN")),
        school         = _esc(student_info.get("school","UNKNOWN")),
        max_marks      = _esc(student_info.get("max_marks","UNKNOWN")),
        obtained_marks = _esc(student_info.get("obtained_marks","UNKNOWN")),
        qa_body        = "\n".join(qa_lines),
    )


def _esc(s: str) -> str:
    """Minimal LaTeX escaping for plain text strings."""
    return (s.replace("&","\\&")
             .replace("%","\\%")
             .replace("$","\\$")
             .replace("#","\\#")
             .replace("_","\\_")
             .replace("{","\\{")
             .replace("}","\\}")
             .replace("~","\\textasciitilde{}")
             .replace("^","\\textasciicircum{}"))


# ==================================================
# MAIN PIPELINE
# ==================================================

def main():
    if len(sys.argv) < 2:
        print(
            "\nUsage:\n"
            "  python3 answer_sheet_ocr.py <answer_sheet.pdf> [question_paper.pdf]\n"
        )
        sys.exit(1)

    answer_pdf_path   = sys.argv[1]
    question_pdf_path = sys.argv[2] if len(sys.argv) >= 3 else None

    stem      = Path(answer_pdf_path).stem
    out_dir   = Path(stem + "_output")
    diag_dir  = out_dir / "diagrams"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("STEP 1 — Rendering answer sheet pages")
    print("="*70)
    answer_images = pdf_to_images(answer_pdf_path)
    print(f"  Pages: {len(answer_images)}")

    # ── Student info (first page) ──────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 2 — Extracting student information")
    print("="*70)
    student_info = extract_student_info(answer_images[0])
    print(json.dumps(student_info, indent=2))

    info_file = out_dir / f"{stem}_student_info.json"
    info_file.write_text(json.dumps(student_info, indent=2, ensure_ascii=False))
    print(f"  Saved → {info_file}")

    # ── OCR every page ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 3 — OCR transcription (all pages)")
    print("="*70)
    page_texts: list[str] = []
    for idx, img in enumerate(answer_images):
        pn = idx + 1
        print(f"  [INFO] OCR Page {pn}/{len(answer_images)}")
        page_texts.append(ocr_page(img, pn))

    raw_transcript = "\n\n".join(
        f"\n\n{'='*60}\nPAGE {i+1}\n{'='*60}\n\n{t}"
        for i, t in enumerate(page_texts)
    )

    # ── Handle diagrams ────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 4 — Processing diagrams / figures")
    print("="*70)
    processed_transcript = extract_diagrams(raw_transcript, diag_dir)
    print(f"  Diagrams saved → {diag_dir}/")

    # ── Parse answers ──────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 5 — Parsing answer blocks")
    print("="*70)
    answers = parse_answers(processed_transcript)
    print(f"  Detected answer blocks: {len(answers)}")

    # ── Extract questions ──────────────────────────────────────────────
    questions: list[dict] = []
    if question_pdf_path:
        print("\n" + "="*70)
        print("STEP 6 — Extracting questions from question paper")
        print("="*70)
        q_images  = pdf_to_images(question_pdf_path)
        questions = extract_questions(q_images)
        print(f"  Questions extracted: {len(questions)}")
        q_file = out_dir / f"{stem}_questions.json"
        q_file.write_text(json.dumps(questions, indent=2, ensure_ascii=False))
        print(f"  Saved → {q_file}")
    else:
        print("\n[INFO] No question paper provided — LaTeX will show answers only.")

    # ── Build LaTeX ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 7 — Generating LaTeX document")
    print("="*70)
    latex_src = build_latex(student_info, questions, answers, processed_transcript)

    tex_file = out_dir / f"{stem}_transcript.tex"
    tex_file.write_text(latex_src, encoding="utf-8")
    print(f"  Saved → {tex_file}")

    # ── Raw transcript ─────────────────────────────────────────────────
    raw_file = out_dir / f"{stem}_raw_transcript.txt"
    raw_file.write_text(processed_transcript, encoding="utf-8")
    print(f"  Raw transcript → {raw_file}")

    print("\n" + "="*70)
    print("✓ DONE")
    print("="*70)
    print(f"\nOutput folder: {out_dir}/")
    print(f"  {stem}_student_info.json   — student metadata")
    print(f"  {stem}_transcript.tex      — LaTeX Q&A document")
    print(f"  {stem}_raw_transcript.txt  — plain text transcript")
    print(f"  diagrams/                  — diagram metadata JSON stubs")
    print()


if __name__ == "__main__":
    main()
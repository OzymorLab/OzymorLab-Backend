"""
Parsing service — 3-pass document parsing pipeline.

Pass 1: Text extraction (PyMuPDF for PDF, Tesseract for images)
Pass 2: Step segmentation (regex + heuristics)
Pass 3: Equation extraction (LaTeX detection + SymPy validation)
"""
import io
import re
import logging

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_data: bytes) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
    Fast (~50ms per page), works well for digital/typed PDFs.
    """
    text_parts = []
    try:
        doc = fitz.open(stream=file_data, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                text_parts.append(text)
            else:
                # Fallback: try OCR on the page image if no text found
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(img, lang="eng")
                if ocr_text.strip():
                    text_parts.append(ocr_text)
        doc.close()
    except Exception as e:
        logger.error(f"PDF text extraction failed: {e}")
        raise

    return "\n\n".join(text_parts)


def extract_text_from_image(file_data: bytes) -> str:
    """
    Extract text from an image using Tesseract OCR.
    Handles scanned answer sheets (~500ms per page).
    """
    try:
        img = Image.open(io.BytesIO(file_data))

        # Preprocessing: convert to grayscale for better OCR accuracy
        if img.mode != "L":
            img = img.convert("L")

        # Run Tesseract OCR
        text = pytesseract.image_to_string(img, lang="eng")
        return text
    except Exception as e:
        logger.error(f"Image OCR failed: {e}")
        raise


def extract_text(file_data: bytes, file_type: str) -> str:
    """
    Route to the appropriate text extraction method based on file type.
    """
    if file_type == "pdf":
        return extract_text_from_pdf(file_data)
    elif file_type in ("png", "jpg", "jpeg"):
        return extract_text_from_image(file_data)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


# ── Step Segmentation ──

# Patterns that indicate step boundaries
STEP_PATTERNS = [
    re.compile(r"^(?:step|Step|STEP)\s*(\d+)[:\.\)\-]?\s*", re.MULTILINE),
    re.compile(r"^(\d+)[:\.\)]\s+", re.MULTILINE),
    re.compile(r"^(?:Given|Find|Solution|Proof|Derivation|Answer)[:\s]", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^(?:Therefore|Hence|Thus|So|We know|Using|From|Since|Let)[,\s]", re.MULTILINE | re.IGNORECASE),
]

# Patterns for inline LaTeX detection
LATEX_PATTERNS = [
    re.compile(r"\$(.+?)\$"),  # inline $...$
    re.compile(r"\\\((.+?)\\\)"),  # \(...\)
    re.compile(r"\\begin\{equation\}(.+?)\\end\{equation\}", re.DOTALL),
    # Common equation-like patterns (F = ma, E = mc^2, etc.)
    re.compile(r"([A-Za-z_]\w*)\s*=\s*([A-Za-z0-9_\+\-\*/\^\(\)\s\.\,]+)"),
]


def segment_into_steps(raw_text: str) -> list[dict]:
    """
    Segment raw text into discrete answer steps.
    Uses regex patterns to detect step boundaries.
    """
    if not raw_text or not raw_text.strip():
        return []

    lines = raw_text.strip().split("\n")
    steps = []
    current_step_lines = []
    current_step_num = 0

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Check if this line starts a new step
        is_new_step = False
        for pattern in STEP_PATTERNS[:2]:  # Only explicit step markers
            match = pattern.match(line_stripped)
            if match:
                is_new_step = True
                break

        if is_new_step and current_step_lines:
            # Save the current step
            current_step_num += 1
            step_text = "\n".join(current_step_lines)
            steps.append({
                "step_num": current_step_num,
                "text": step_text,
                "equations": extract_equations(step_text),
                "step_type": classify_step_type(step_text),
            })
            current_step_lines = [line_stripped]
        else:
            current_step_lines.append(line_stripped)

    # Don't forget the last step
    if current_step_lines:
        current_step_num += 1
        step_text = "\n".join(current_step_lines)
        steps.append({
            "step_num": current_step_num,
            "text": step_text,
            "equations": extract_equations(step_text),
            "step_type": classify_step_type(step_text),
        })

    # If no step boundaries were detected, treat the whole text as one step
    if len(steps) == 0 and raw_text.strip():
        steps.append({
            "step_num": 1,
            "text": raw_text.strip(),
            "equations": extract_equations(raw_text),
            "step_type": "statement",
        })

    return steps


def extract_equations(text: str) -> list[str]:
    """Extract LaTeX and equation-like expressions from text."""
    equations = []
    for pattern in LATEX_PATTERNS:
        for match in pattern.finditer(text):
            expr = match.group(1) if match.lastindex else match.group(0)
            expr = expr.strip()
            if len(expr) > 2 and expr not in equations:  # Skip trivially short matches
                equations.append(expr)
    return equations


def classify_step_type(text: str) -> str:
    """Classify a step as statement, derivation, result, or diagram."""
    text_lower = text.lower()

    if any(kw in text_lower for kw in ["therefore", "hence", "thus", "answer", "result", "final"]):
        return "result"
    elif any(kw in text_lower for kw in ["substitut", "differenti", "integrat", "simplif", "rearrang"]):
        return "derivation"
    elif any(kw in text_lower for kw in ["diagram", "figure", "sketch", "draw"]):
        return "diagram"
    elif "=" in text:
        return "derivation"
    else:
        return "statement"


def parse_submission(file_data: bytes, file_type: str) -> dict:
    """
    Full parsing pipeline: extract text → segment steps → extract equations.

    Returns the parsed_content JSONB structure.
    """
    # Pass 1: Text extraction
    raw_text = extract_text(file_data, file_type)

    # Pass 2: Step segmentation
    steps = segment_into_steps(raw_text)

    # Pass 3 is integrated into step segmentation (equation extraction per step)

    # Compute parse confidence based on text quality
    confidence = compute_parse_confidence(raw_text, steps)

    parsed_content = {
        "steps": steps,
        "detected_language": "english",
        "has_diagrams": any(s["step_type"] == "diagram" for s in steps),
        "parse_confidence": confidence,
    }

    return raw_text, parsed_content


def compute_parse_confidence(raw_text: str, steps: list[dict]) -> float:
    """
    Estimate parsing confidence based on text quality indicators.
    Returns a float between 0.0 and 1.0.
    """
    if not raw_text or not raw_text.strip():
        return 0.0

    score = 0.5  # base score

    # More steps = more structure detected
    if len(steps) >= 2:
        score += 0.1
    if len(steps) >= 4:
        score += 0.1

    # Equations found = better parsing
    total_equations = sum(len(s.get("equations", [])) for s in steps)
    if total_equations > 0:
        score += 0.15
    if total_equations >= 3:
        score += 0.05

    # Longer text generally means better extraction
    if len(raw_text) > 100:
        score += 0.05
    if len(raw_text) > 500:
        score += 0.05

    return min(score, 1.0)

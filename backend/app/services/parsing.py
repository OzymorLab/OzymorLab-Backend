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
from app.services.llm_client import extract_text_from_image_gemini

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_data: bytes) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
    Fast (~50ms per page) for digital/typed PDFs.

    For scanned PDFs (pages with no extractable text), renders the page
    as an image and falls back to Gemini Vision OCR for transcription.
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
                # ── Scanned PDF fallback: render page as image → Gemini Vision OCR ──
                logger.info(
                    f"Page {page_num + 1} has no extractable text. "
                    f"Rendering as image for Gemini Vision OCR fallback."
                )
                try:
                    # Render page at 2x resolution for better OCR accuracy
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                    image_bytes = pix.tobytes("png")
                    ocr_text = extract_text_from_image_gemini(image_bytes)
                    if ocr_text and ocr_text.strip():
                        text_parts.append(ocr_text)
                        logger.info(
                            f"Page {page_num + 1} OCR fallback extracted "
                            f"{len(ocr_text)} characters."
                        )
                    else:
                        logger.warning(
                            f"Page {page_num + 1} OCR fallback returned empty text."
                        )
                except Exception as ocr_err:
                    logger.error(
                        f"Gemini Vision OCR fallback failed for page {page_num + 1}: {ocr_err}"
                    )
        doc.close()
    except Exception as e:
        logger.error(f"PDF text extraction failed: {e}")
        raise

    return "\n\n".join(text_parts)


def extract_text_from_image(file_data: bytes) -> str:
    """
    Extract text from an image using Gemini 2.5 Pro Vision.
    Extremely accurate for handwritten math and fuzzy scans.
    """
    try:
        # Run Gemini Vision OCR
        text = extract_text_from_image_gemini(file_data)
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

# Patterns for strict hierarchical question markers
QUESTION_MARKERS = [
    re.compile(r"^(?:section|part|group)\s*([a-z0-9])", re.IGNORECASE),
    re.compile(r"^(?:ans|q|question|answer)?\s*(\d+)\s*(?:\(([a-z])\))?", re.IGNORECASE),
    re.compile(r"^\(([a-zivx]+)\)", re.IGNORECASE),  # e.g., (b) or (ii)
]

class AnswerSheetState:
    """Tracks context across text blocks to resolve implicit breadcrumbs."""
    def __init__(self, question_schema: dict | None = None):
        self.current_section = None
        self.current_question = None
        self.current_subquestion = None
        self.schema = question_schema or {}

    def update_from_text(self, text: str) -> bool:
        """Parse text for markers and update state. Returns True if state mutated."""
        text_lower = text.strip().lower()
        mutated = False
        
        # Check Section
        sec_match = QUESTION_MARKERS[0].match(text_lower)
        if sec_match:
            self.current_section = sec_match.group(1).upper()
            self.current_question = None
            self.current_subquestion = None
            mutated = True
            
        # Check Question (e.g. Q2, Ans 3, 4(a))
        q_match = QUESTION_MARKERS[1].match(text_lower)
        if q_match:
            q_num = q_match.group(1)
            sub_q = q_match.group(2)
            if q_num:
                self.current_question = q_num
                self.current_subquestion = sub_q.upper() if sub_q else None
                mutated = True
                
        # Check bare subquestion (e.g. (b))
        sub_match = QUESTION_MARKERS[2].match(text_lower)
        if sub_match and not q_match and self.current_question:
            # Inherit current question!
            self.current_subquestion = sub_match.group(1).upper()
            mutated = True

        return mutated

    def get_state_string(self) -> str:
        parts = []
        if self.current_section: parts.append(f"Sec {self.current_section}")
        if self.current_question:
            q_str = f"Q{self.current_question}"
            if self.current_subquestion: q_str += f"({self.current_subquestion})"
            parts.append(q_str)
        return " | ".join(parts) if parts else "Unknown Context"

# Patterns for inline LaTeX detection
LATEX_PATTERNS = [
    re.compile(r"\$(.+?)\$"),  # inline $...$
    re.compile(r"\\\((.+?)\\\)"),  # \(...\)
    re.compile(r"\\begin\{equation\}(.+?)\\end\{equation\}", re.DOTALL),
    # Common equation-like patterns (F = ma, E = mc^2, etc.)
    re.compile(r"([A-Za-z_]\w*)\s*=\s*([A-Za-z0-9_\+\-\*/\^\(\)\s\.\,]+)"),
]


def segment_into_steps(raw_text: str, question_schema: dict | None = None) -> list[dict]:
    """
    Segment raw text into discrete answer steps using a Stateful Tracker.
    """
    if not raw_text or not raw_text.strip():
        return []

    lines = raw_text.strip().split("\n")
    steps = []
    current_step_lines = []
    current_step_num = 0
    state_tracker = AnswerSheetState(question_schema)

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Check for context updates (Q1, Section A, etc.)
        state_mutated = state_tracker.update_from_text(line_stripped)

        # Check if this line starts a new logical step
        is_new_step = False
        for pattern in STEP_PATTERNS[:2]:  # Explicit step markers
            if pattern.match(line_stripped):
                is_new_step = True
                break

        # If context changed drastically, or explicitly new step
        if (is_new_step or state_mutated) and current_step_lines:
            current_step_num += 1
            step_text = "\n".join(current_step_lines)
            steps.append({
                "step_num": current_step_num,
                "text": step_text,
                "equations": extract_equations(step_text),
                "step_type": classify_step_type(step_text),
                "context": state_tracker.get_state_string()
            })
            current_step_lines = [line_stripped]
        else:
            current_step_lines.append(line_stripped)

    # Flush remaining
    if current_step_lines:
        current_step_num += 1
        step_text = "\n".join(current_step_lines)
        steps.append({
            "step_num": current_step_num,
            "text": step_text,
            "equations": extract_equations(step_text),
            "step_type": classify_step_type(step_text),
            "context": state_tracker.get_state_string()
        })

    # If no boundaries detected
    if len(steps) == 0 and raw_text.strip():
        state_tracker.update_from_text(raw_text.strip())
        steps.append({
            "step_num": 1,
            "text": raw_text.strip(),
            "equations": extract_equations(raw_text),
            "step_type": "statement",
            "context": state_tracker.get_state_string()
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

    # Pass 2: Step segmentation with Stateful Tracking
    steps = segment_into_steps(raw_text)

    # Pass 3 is integrated into step segmentation (equation extraction per step)

    # Compute parse confidence based on text quality
    confidence = compute_parse_confidence(raw_text, steps)

    # Map out context boundaries to ensure LLM grading sees context
    has_orphaned_blocks = any(s.get("context") == "Unknown Context" for s in steps)
    if has_orphaned_blocks:
        logger.warning("Submission has orphaned text blocks lacking question mapping context. Tier 2 semantic fallback may be required.")

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

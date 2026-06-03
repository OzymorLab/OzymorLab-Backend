"""
Parsing service — page-by-page OCR, diagram cropping, and question alignment.
"""
import io
import re
import logging
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ─── OCR & DIAGRAM EXTRACTION ──────────────────────────────────────────────────

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

def _render_page_to_png(page: fitz.Page, scale: float = 2.0) -> bytes:
    """Render a fitz page at `scale`x resolution and return PNG bytes."""
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    return pix.tobytes("png")


def _ocr_page_single(image_bytes: bytes, page_num: int) -> str:
    """Run OCR on a single page image using Gemini Client."""
    from app.services.llm_client import get_client
    from app.config import settings
    from google.genai import types as genai_types

    client = get_client()
    try:
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[
                genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                OCR_PROMPT,
            ],
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=4096,
            ),
        )
        return response.text or ""
    except Exception as e:
        logger.error(f"[OCR] Page {page_num} Gemini call failed: {e}")
        return ""


def detect_diagram_boxes(image_bytes: bytes, mime_type: str) -> list[dict]:
    """
    Call Gemini to detect diagrams, tables, flowcharts and return their normalized
    bounding boxes [ymin, xmin, ymax, xmax] in the range 0 to 1000.
    """
    from app.services.llm_client import get_client, parse_json_response
    from app.config import settings
    from google.genai import types as genai_types

    prompt = """
    Identify all diagrams, hand-drawn figures, drawings, graphs, tables, maps, circuits, or flowcharts on this page.
    For each detected diagram/element, return its bounding box as normalized coordinates [ymin, xmin, ymax, xmax] in the range 0 to 1000.
    Return ONLY a valid JSON list of objects, with keys "type" and "box_2d". Example:
    [
      {"type": "biological_diagram", "box_2d": [100, 200, 500, 800]}
    ]
    """
    client = get_client()
    try:
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[
                genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt,
            ],
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1024,
            ),
        )
        raw_text = response.text or ""
        parsed = parse_json_response(raw_text)
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception as e:
        logger.error(f"[Parse] Diagram detection failed: {e}")
        return []


def crop_diagram(image_bytes: bytes, box_2d: list[int]) -> bytes:
    """
    Crop the image bytes using the normalized box coordinates [ymin, xmin, ymax, xmax] (0 to 1000).
    Returns cropped PNG bytes.
    """
    from PIL import Image
    import io
    try:
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        ymin, xmin, ymax, xmax = box_2d

        left = int(xmin * width / 1000)
        top = int(ymin * height / 1000)
        right = int(xmax * width / 1000)
        bottom = int(ymax * height / 1000)

        # Add a tiny margin
        margin = 15
        left = max(0, left - margin)
        top = max(0, top - margin)
        right = min(width, right + margin)
        bottom = min(height, bottom + margin)

        cropped_img = img.crop((left, top, right, bottom))
        out_bytes = io.BytesIO()
        cropped_img.save(out_bytes, format="PNG")
        return out_bytes.getvalue()
    except Exception as e:
        logger.error(f"[Parse] Cropping diagram failed: {e}")
        return b""


def process_page_diagrams(
    ocr_text: str,
    page_img_bytes: bytes,
    page_num: int,
    folder_id: str,
) -> tuple[str, list[dict]]:
    r"""
    Detect diagrams on the page, crop them, upload to Supabase, and replace
    [DIAGRAM_START]...[DIAGRAM_END] placeholders with LaTeX \includegraphics blocks.
    Returns (processed_ocr_text, list_of_cropped_diagram_info).
    """
    from app.services.ingestion import upload_file

    # 1. Detect diagram bounding boxes
    boxes = detect_diagram_boxes(page_img_bytes, "image/png")
    logger.info(f"[Parse] Page {page_num}: Detected {len(boxes)} visual box(es)")

    cropped_diagrams = []
    for idx, box_info in enumerate(boxes):
        box = box_info.get("box_2d")
        dtype = box_info.get("type", "diagram")
        if not box or len(box) != 4:
            continue

        # Crop the image
        cropped_bytes = crop_diagram(page_img_bytes, box)
        if not cropped_bytes:
            continue

        # Upload to Supabase Storage
        filename = f"diagram_{page_num}_{idx}.png"
        folder = f"submissions/{folder_id}"
        try:
            key = upload_file(cropped_bytes, filename, "image/png", folder=folder)
            cropped_diagrams.append({
                "type": dtype,
                "box": box,
                "key": key,
                "filename": filename
            })
            logger.info(f"[Parse] Cropped diagram uploaded: {key}")
        except Exception as e:
            logger.error(f"[Parse] Failed to upload cropped diagram {filename}: {e}")

    # 2. Match text blocks to cropped diagrams
    pattern = re.compile(r'\[DIAGRAM_START\](.*?)\[DIAGRAM_END\]', re.DOTALL | re.IGNORECASE)
    block_counter = [0]

    def replace_block(match: re.Match) -> str:
        idx = block_counter[0]
        block_counter[0] += 1

        block_content = match.group(1).strip()
        fields: dict[str, str] = {}
        for line in block_content.splitlines():
            if ':' in line:
                key, _, val = line.partition(':')
                fields[key.strip().upper()] = val.strip()

        dtype = fields.get("TYPE", "other")
        desc = fields.get("DESCRIPTION", "")
        labels = fields.get("LABELS", "")

        # If we have a matching cropped diagram
        if idx < len(cropped_diagrams):
            diag = cropped_diagrams[idx]
            # Use diagram S3 key reference inside LaTeX
            latex_block = f"""
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.8\\textwidth]{{{diag['key']}}}
\\caption{{{desc} (Labels: {labels})}}
\\end{{figure}}
"""
        else:
            # Fallback placeholder
            latex_block = f"""
\\begin{{figure}}[H]
\\centering
\\fbox{{\\parbox{{0.8\\textwidth}}{{
  \\textbf{{[Hand-drawn {dtype}]}}\\\\
  \\textit{{{desc}}}\\\\
  \\textbf{{Labels:}} {labels}
}}}}
\\caption{{{desc}}}
\\end{{figure}}
"""
        return latex_block.strip()

    processed_text = pattern.sub(replace_block, ocr_text)
    return processed_text, cropped_diagrams


# ─── Heuristic & LLM Question-Answer Alignment ─────────────────────────────

def parse_answers_fallback(full_transcript: str) -> dict[str, str]:
    """
    Heuristic: find lines like "1.", "Q1", "1)", "(1)" etc. and group
    following text as that question's answer.
    Returns {question_number_str: answer_text}.
    """
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


def align_answers_to_questions(
    full_transcript: str,
    questions: list[dict],
) -> dict[str, str]:
    """
    Call Gemini to align student answer blocks from the transcript to the
    list of questions from the task rubric.
    """
    from app.services.llm_client import get_client, parse_json_response
    from app.config import settings
    from google.genai import types as genai_types

    if not questions:
        return {}

    questions_list_str = "\n".join(
        f"- Question/Step {q.get('step_num')}: {q.get('description')} (Max Marks: {q.get('marks')})"
        for q in questions
    )

    prompt = f"""
    You are given:
    1. A list of exam questions/rubric steps:
    {questions_list_str}

    2. The full transcript of a student's answer sheet:
    \"\"\"
    {full_transcript}
    \"\"\"

    Your task is to map each question to the student's corresponding answer from the transcript.
    Extract the answer text verbatim from the transcript, including any LaTeX equations and diagram figure/caption placeholders.
    Do not add any comments or grade/solve the question. Just extract the student's written answer.
    If a question was not answered, map it to an empty string "".

    Return ONLY a JSON object mapping the question/step number (as a string) to the student's answer block.
    No other text, no markdown code fences. Example:
    {{
      "1": "First answer...",
      "2": "Second answer..."
    }}
    """

    client = get_client()
    try:
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[prompt],
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=4096,
            ),
        )
        raw_text = response.text or ""
        parsed = parse_json_response(raw_text)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
        return {}
    except Exception as e:
        logger.error(f"[Parse] LLM alignment failed: {e}")
        return parse_answers_fallback(full_transcript)


# ─── Main parse entry point ───────────────────────────────────────────────────

def extract_text(file_data: bytes, file_type: str) -> str:
    """Route to the appropriate text extraction method based on file type."""
    if file_type == "pdf":
        return extract_text_from_pdf(file_data)
    elif file_type in ("png", "jpg", "jpeg", "webp"):
        return extract_text_from_image(file_data)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def extract_text_from_pdf(file_data: bytes) -> str:
    """Render PDF page-by-page and run OCR."""
    doc = fitz.open(stream=file_data, filetype="pdf")
    pages_text = []
    for page_num in range(len(doc)):
        img_bytes = _render_page_to_png(doc[page_num])
        ocr_text = _ocr_page_single(img_bytes, page_num + 1)
        pages_text.append(ocr_text)
    doc.close()
    return "\n\n".join(pages_text)


def extract_text_from_image(file_data: bytes) -> str:
    """OCR single image."""
    return _ocr_page_single(file_data, 1)


LATEX_PATTERNS = [
    re.compile(r"\$(.+?)\$"),
    re.compile(r"\\\((.+?)\\\)"),
    re.compile(r"\\begin\{equation\}(.+?)\\end\{equation\}", re.DOTALL),
    re.compile(r"([A-Za-z_]\w*)\s*=\s*([A-Za-z0-9_\+\-\*/\^\(\)\s\.\,]+)"),
]

def extract_equations(text: str) -> list[str]:
    equations: list[str] = []
    for pattern in LATEX_PATTERNS:
        for match in pattern.finditer(text):
            expr = match.group(1) if match.lastindex else match.group(0)
            expr = expr.strip()
            if len(expr) > 2 and expr not in equations:
                equations.append(expr)
    return equations


def parse_submission(
    file_data: bytes,
    file_type: str,
    submission_id: str | None = None,
    questions: list[dict] | None = None,
) -> tuple[str, dict]:
    """
    Refactored page-by-page OCR, diagram extraction, and Q&A alignment pipeline.
    """
    import uuid
    from app.services.ingestion import upload_file

    folder_id = submission_id or str(uuid.uuid4())
    page_images = []

    # 1. Render/load pages
    if file_type.lower() == "pdf":
        doc = fitz.open(stream=file_data, filetype="pdf")
        for i in range(len(doc)):
            page_images.append(_render_page_to_png(doc[i]))
        doc.close()
    else:
        page_images.append(file_data)

    logger.info(f"[Parse] Processing submission {folder_id} with {len(page_images)} page(s) page-by-page")

    # 2. OCR and diagram extraction page-by-page
    pages_data = []
    all_cropped_diagrams = []
    full_transcript_parts = []

    for page_idx, img_bytes in enumerate(page_images):
        page_num = page_idx + 1
        logger.info(f"[Parse] Page {page_num}: transcribing & detecting diagrams")

        # Upload page image as intermediate file
        page_img_key = f"page_images/page_{page_num}.png"
        try:
            page_img_key = upload_file(img_bytes, f"page_{page_num}.png", "image/png", folder=f"submissions/{folder_id}")
        except Exception as e:
            logger.warning(f"[Parse] Page {page_num} image upload failed: {e}")

        # OCR transcription
        ocr_text = _ocr_page_single(img_bytes, page_num)

        # Save raw transcript as intermediate file
        try:
            upload_file(ocr_text.encode("utf-8"), f"raw_transcript_{page_num}.txt", "text/plain", folder=f"submissions/{folder_id}")
        except Exception as e:
            logger.warning(f"[Parse] Page {page_num} transcript upload failed: {e}")

        # Diagram detection and cropping
        processed_ocr_text, page_diagrams = process_page_diagrams(ocr_text, img_bytes, page_num, folder_id)
        all_cropped_diagrams.extend(page_diagrams)

        pages_data.append({
            "page_num": page_num,
            "image_key": page_img_key,
            "ocr_text": processed_ocr_text
        })

        full_transcript_parts.append(
            f"\n\n{'='*60}\nPAGE {page_num}\n{'='*60}\n\n{processed_ocr_text}"
        )

    full_transcript = "\n\n".join(full_transcript_parts)

    # 3. Align answers to questions
    aligned_answers = {}
    if questions:
        logger.info(f"[Parse] Aligning transcript to {len(questions)} rubric question(s)")
        aligned_answers = align_answers_to_questions(full_transcript, questions)
    else:
        logger.info("[Parse] No rubric questions provided, falling back to heuristic parsing")
        aligned_answers = parse_answers_fallback(full_transcript)

    # 4. Form steps matching the questions or parsed blocks
    steps = []
    # If questions provided, align with them
    if questions:
        for q in questions:
            q_num_str = str(q.get("step_num"))
            ans_text = aligned_answers.get(q_num_str, "")
            
            # Find diagrams used in this answer text
            diagrams_in_step = []
            for d in all_cropped_diagrams:
                if d["key"] in ans_text:
                    diagrams_in_step.append(d)

            steps.append({
                "step_num": q.get("step_num", 1),
                "step_type": q.get("component_type", q.get("step_type", "statement")),
                "text": ans_text,
                "equations": extract_equations(ans_text),
                "diagrams": diagrams_in_step,
            })
    else:
        # Fallback step creation
        for q_num_str, ans_text in aligned_answers.items():
            try:
                s_num = int(q_num_str)
            except ValueError:
                s_num = len(steps) + 1
            steps.append({
                "step_num": s_num,
                "step_type": "statement",
                "text": ans_text,
                "equations": extract_equations(ans_text),
                "diagrams": []
            })

    # 5. Compute parse confidence
    confidence = compute_parse_confidence(full_transcript, steps)

    parsed_content = {
        "steps": steps,
        "detected_language": "english",
        "has_diagrams": len(all_cropped_diagrams) > 0,
        "parse_confidence": confidence,
        "pages": pages_data,
        "cropped_diagrams": all_cropped_diagrams
    }

    return full_transcript, parsed_content


class AnswerSheetState:
    """Stateful tracker for contextual breadcrumbs in document parsing."""
    
    def __init__(self):
        """Initialize with unknown context."""
        self.section: str | None = None
        self.question: str | None = None
        self.subquestion: str | None = None
    
    def update_from_text(self, text: str) -> bool:
        """
        Update state based on text patterns.
        Returns True if state was mutated, False otherwise.
        """
        initial_state = (self.section, self.question, self.subquestion)
        
        # Match section patterns: "Section A", "Part C", etc.
        section_match = re.search(r'(?:Section|Part)\s+([A-Z])', text, re.IGNORECASE)
        if section_match:
            self.section = section_match.group(1)
            self.question = None
            self.subquestion = None
            return initial_state != (self.section, self.question, self.subquestion)
        
        # Match question patterns: "Question 2", "Q2", etc.
        question_match = re.search(r'(?:Question|Q\.?)\s*(\d+)', text, re.IGNORECASE)
        if question_match:
            self.question = question_match.group(1)
            self.subquestion = None
            return initial_state != (self.section, self.question, self.subquestion)
        
        # Match subquestion patterns: "(b)", "(ii)", etc.
        subquestion_match = re.search(r'\(([a-z]|[ivx]+)\)', text, re.IGNORECASE)
        if subquestion_match:
            self.subquestion = subquestion_match.group(1).upper()
            return initial_state != (self.section, self.question, self.subquestion)
        
        return False
    
    def get_state_string(self) -> str:
        """Return formatted state string."""
        if self.section is None:
            return "Unknown Context"
        
        parts = [f"Sec {self.section}"]
        if self.question is not None:
            parts.append(f"Q{self.question}")
            if self.subquestion is not None:
                parts[-1] += f"({self.subquestion})"
        
        return " | ".join(parts)


def segment_into_steps(raw_text: str) -> list[dict]:
    """
    Segment raw text into discrete structured steps with contexts.
    Returns list of dicts with 'text' key.
    """
    # Split by common step markers, capturing the marker and following text
    step_pattern = re.compile(
        r'((?:^|\n)\s*(?:Step\s+\d+|Section\s+[A-Z]|Question\s+\d+|Part\s+[A-Z]|[0-9]+\.).*?)(?=(?:\n\s*(?:Step|Section|Question|Part|[0-9]+\.))|$)',
        re.MULTILINE | re.IGNORECASE | re.DOTALL
    )
    
    segments = []
    
    # First try to match structured patterns
    matches = list(step_pattern.finditer(raw_text))
    
    if matches:
        for match in matches:
            text = match.group(0).strip()
            if text:
                segments.append({"text": text})
    
    # If no structured patterns found, split by newlines
    if not segments:
        lines = raw_text.split('\n')
        current_segment = []
        for line in lines:
            stripped = line.strip()
            if stripped:
                current_segment.append(stripped)
            elif current_segment:
                segments.append({"text": '\n'.join(current_segment)})
                current_segment = []
        
        if current_segment:
            segments.append({"text": '\n'.join(current_segment)})
    
    return segments if segments else [{"text": raw_text}]


def classify_step_type(text: str) -> str:
    """
    Classify the type of step based on heuristic keyword matching.
    Returns one of: "result", "derivation", "diagram", "statement"
    """
    text_lower = text.lower()
    
    # Check for result indicators
    if any(word in text_lower for word in ["therefore", "hence", "thus", "answer is", "the answer"]):
        return "result"
    
    # Check for derivation indicators
    if any(word in text_lower for word in ["integrate", "derive", "differentiate", "apply", "using", "formula"]):
        return "derivation"
    
    # Check for diagram indicators
    if any(word in text_lower for word in ["figure", "diagram", "graph", "chart", "table", "refer"]):
        return "diagram"
    
    # Default to statement
    return "statement"


def compute_parse_confidence(raw_text: str, steps: list[dict]) -> float:
    if not raw_text or not raw_text.strip():
        return 0.0
    score = 0.5
    if len(steps) >= 2:
        score += 0.1
    if len(steps) >= 4:
        score += 0.1
    total_eq = sum(len(s.get("equations", [])) for s in steps)
    if total_eq > 0:
        score += 0.15
    if total_eq >= 3:
        score += 0.05
    if len(raw_text) > 100:
        score += 0.05
    if len(raw_text) > 500:
        score += 0.05
    return min(score, 1.0)

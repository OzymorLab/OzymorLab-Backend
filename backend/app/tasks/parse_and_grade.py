"""
parse_and_grade — end-to-end pipeline for a single answer sheet.

Phase 1: OCR + diagram crop + Q&A alignment → PARSED
Phase 2: Grade against approved rubric → GRADED + SubmissionStep rows
"""
import asyncio
import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ─── LaTeX transcript builder (inline — no debug_pdf import) ─────────────────

def _esc(s: str) -> str:
    """Minimal LaTeX escaping for plain text strings."""
    for old, new in [
        ("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
        ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
        ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
    ]:
        s = s.replace(old, new)
    return s


def _build_latex_transcript(
    student_info: dict,
    rubric_steps: list,
    answers_map: dict,          # step_num_str -> answer_text
    step_grades: list,          # list of step_grade dicts from grading result
    full_transcript: str,
) -> str:
    """Build a human-readable LaTeX transcript pairing questions with answers and scores."""
    import re

    TEMPLATE = r"""
\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{fancyhdr}
\geometry{margin=2.5cm}
\pagestyle{fancy}
\fancyhf{}
\rhead{%(subject)s}
\lhead{Answer Sheet Transcript}
\cfoot{\thepage}
\begin{document}
\begin{center}
  {\LARGE\bfseries Answer Sheet Transcript}\\[6pt]
  {\large\itshape %(subject)s}
\end{center}
\vspace{8pt}
\begin{center}
\begin{tabular}{|l|l||l|l|}
\hline
\textbf{Student} & %(name)s & \textbf{Roll No.} & %(roll)s \\
\hline
\textbf{Class} & %(cls)s & \textbf{Date} & %(date)s \\
\hline
\textbf{Max Marks} & %(max_marks)s & \textbf{Obtained} & %(obtained)s \\
\hline
\end{tabular}
\end{center}
\vspace{12pt}
\hrule
\vspace{12pt}
%(qa_body)s
\end{document}
""".strip()

    grade_map = {str(sg.get("step_num")): sg for sg in step_grades}

    qa_lines = []
    for r_step in rubric_steps:
        sn = str(r_step.get("step_num", "?"))
        q_text = r_step.get("description", "")
        marks = r_step.get("marks", 0)
        ans_text = answers_map.get(sn, "")
        sg = grade_map.get(sn, {})
        awarded = sg.get("marks_awarded", "—")
        justification = sg.get("justification", "")

        block = f"\\subsection*{{Question {sn}  [{marks} marks]}}\n"
        block += f"\\noindent\\fbox{{\\begin{{minipage}}{{0.97\\textwidth}}\n"
        block += f"\\textbf{{Q{sn}.}} {q_text}\n"
        block += f"\\end{{minipage}}}}\n\n"
        block += f"\\textbf{{Answer:}}\n\n"
        block += (ans_text.strip() if ans_text.strip() else "\\textit{[No answer detected]}")
        block += f"\n\n\\medskip\n"
        block += f"\\textbf{{Score: }} {awarded} / {marks}"
        if justification:
            safe_j = justification.replace("%", r"\%").replace("&", r"\&")[:300]
            block += f"\n\n\\textit{{\\small {safe_j}}}"
        block += "\n\n\\bigskip\\hrule\\bigskip\n"
        qa_lines.append(block)

    if not qa_lines:
        qa_lines = ["\\section*{Full Transcript}\n\n" + full_transcript]

    return TEMPLATE % {
        "subject":    _esc(student_info.get("subject", "Unknown")),
        "name":       _esc(student_info.get("name", "Unknown")),
        "roll":       _esc(student_info.get("roll_number", "—")),
        "cls":        _esc(student_info.get("class", "—")),
        "date":       _esc(student_info.get("date", "—")),
        "max_marks":  _esc(str(student_info.get("max_marks", "—"))),
        "obtained":   _esc(str(student_info.get("obtained_marks", "—"))),
        "qa_body":    "\n".join(qa_lines),
    }


# ─── Main pipeline ────────────────────────────────────────────────────────────

async def parse_and_grade(submission_id: str) -> None:
    """
    Full end-to-end pipeline for one answer sheet.
    Safe to call as a fire-and-forget asyncio task.

    Phase 1 — OCR + diagram crop + alignment → PARSED
    Phase 2 — Grade against approved rubric   → GRADED + SubmissionStep rows + LaTeX transcript
    """
    from app.db.session import async_session_factory
    from app.db.models import (
        Submission, TaskRubric, GradingRun, GradeResult, Task, SubmissionStep
    )
    from app.services.ingestion import download_file, upload_file
    from app.services.parsing import parse_submission as _parse_sync
    from app.services.grading import grade_submission as _grade_sync
    from app.config import settings
    from sqlalchemy import select, delete
    from sqlalchemy.orm import attributes

    sub_uuid = uuid.UUID(submission_id)

    # ── PHASE 1: Parse ────────────────────────────────────────────────────────
    async with async_session_factory() as session:
        try:
            result = await session.execute(
                select(Submission).filter_by(id=sub_uuid)
            )
            submission = result.scalar_one_or_none()
            if not submission:
                logger.error(f"[P&G] Submission {submission_id} not found")
                return

            file_key  = submission.file_key
            file_type = submission.file_type or "pdf"
            task_id   = submission.task_id

            submission.status = "PARSING"
            await session.commit()
            logger.info(f"[P&G] {submission_id} → PARSING")

            # Fetch rubric questions to guide alignment (any active rubric, approval not required yet)
            rubric_result = await session.execute(
                select(TaskRubric)
                .filter_by(task_id=task_id, is_active=True)
                .order_by(TaskRubric.created_at.desc())
                .limit(1)
            )
            rubric_for_parse = rubric_result.scalar_one_or_none()
            questions = None
            if rubric_for_parse:
                questions = list(rubric_for_parse.rubric_json.get("steps", []))

            # All sync/blocking work offloaded to thread pool
            file_data = await asyncio.to_thread(download_file, file_key)
            raw_text, parsed_content = await asyncio.to_thread(
                _parse_sync, file_data, file_type, str(submission.id), questions
            )

            submission.raw_text       = raw_text
            submission.parsed_content = parsed_content
            submission.status         = "PARSED"
            submission.error_message  = None
            await session.commit()
            logger.info(
                f"[P&G] {submission_id} → PARSED "
                f"({len(parsed_content.get('steps', []))} steps)"
            )

        except Exception as e:
            logger.error(f"[P&G] Parse failed for {submission_id}: {e}", exc_info=True)
            try:
                result = await session.execute(
                    select(Submission).filter_by(id=sub_uuid)
                )
                sub = result.scalar_one_or_none()
                if sub:
                    sub.status        = "FAILED"
                    sub.error_message = str(e)
                    await session.commit()
            except Exception:
                await session.rollback()
            return   # Stop — can't grade if parsing failed

    # ── PHASE 2: Auto-grade if approved rubric exists ─────────────────────────
    async with async_session_factory() as session:
        try:
            result = await session.execute(
                select(Submission).filter_by(id=sub_uuid)
            )
            submission = result.scalar_one_or_none()

            if not submission or submission.status != "PARSED":
                logger.info(
                    f"[P&G] {submission_id} not in PARSED state "
                    f"(got '{getattr(submission, 'status', 'none')}') — skipping auto-grade"
                )
                return

            # Need an APPROVED rubric to grade
            rubric_result = await session.execute(
                select(TaskRubric)
                .filter_by(task_id=task_id, is_active=True, approval_status="APPROVED")
                .order_by(TaskRubric.created_at.desc())
                .limit(1)
            )
            rubric = rubric_result.scalar_one_or_none()

            if not rubric:
                logger.info(
                    f"[P&G] {submission_id} parsed — no APPROVED rubric yet, "
                    "leaving as PARSED for manual grading"
                )
                return

            task_result = await session.execute(
                select(Task).filter_by(id=task_id)
            )
            task = task_result.scalar_one_or_none()

            run = GradingRun(
                id=uuid.uuid4(),
                task_id=task_id,
                rubric_version=rubric.version,
                model=settings.GEMINI_MODEL,
                temperature=settings.GRADING_TEMPERATURE,
                description="Auto-grade on upload",
                status="RUNNING",
                total_submissions=1,
                graded_count=0,
                failed_count=0,
                created_by=submission.student_id,
            )
            session.add(run)
            await session.flush()
            await session.refresh(run)

            submission.status = "GRADING"
            await session.commit()
            logger.info(
                f"[P&G] {submission_id} → GRADING "
                f"(run={run.id}, rubric_v{rubric.version})"
            )

            rubric_data = dict(rubric.rubric_json)
            rubric_data["grading_notes"] = rubric.grading_notes or ""
            rubric_data["model"]         = settings.GEMINI_MODEL
            rubric_data["max_marks"]     = float(task.max_marks) if task and task.max_marks else 0

            result_data = await asyncio.to_thread(
                _grade_sync,
                rubric_data,
                submission.parsed_content,
                settings.GRADING_TEMPERATURE,
                task.subject     if task else "General",
                task.board       if task else "Generic",
                task.grade_level if task else "Unknown",
                submission.file_key,
                str(submission.id),
                None,
            )

            # ── Persist SubmissionStep rows ───────────────────────────────────
            await session.execute(
                delete(SubmissionStep).filter_by(submission_id=submission.id)
            )

            rubric_steps  = rubric_data.get("steps", [])
            parsed_steps  = submission.parsed_content.get("steps", [])
            # Build map by step_num for quick lookup
            parsed_map    = {str(ps.get("step_num")): ps for ps in parsed_steps}

            for sg in result_data["step_grades"]:
                s_num_str = str(sg["step_num"])
                p_step = parsed_map.get(s_num_str)

                step_text  = p_step.get("text", "")           if p_step else ""
                step_latex = p_step.get("latex") or p_step.get("text", "") if p_step else ""
                bbox_data  = None
                if p_step:
                    diagrams = p_step.get("diagrams", [])
                    if diagrams:
                        bbox_data = {
                            "diagram_key":      diagrams[0].get("key"),
                            "diagram_filename": diagrams[0].get("filename"),
                            "box":              diagrams[0].get("box"),
                        }

                r_step_meta = next(
                    (r for r in rubric_steps if str(r.get("step_num")) == s_num_str), {}
                )

                session.add(SubmissionStep(
                    submission_id=submission.id,
                    step_num=sg["step_num"],
                    step_type=(
                        sg.get("step_type")
                        or r_step_meta.get("component_type", "statement")
                    ),
                    text=step_text,
                    latex=step_latex,
                    marks_awarded=sg["marks_awarded"],
                    max_marks=sg["max_marks"],
                    justification=sg["justification"],
                    error_type=sg.get("error_type"),
                    bounding_box=bbox_data,
                ))

            # ── Build LaTeX transcript and store it ───────────────────────────
            try:
                answers_map = {
                    str(ps.get("step_num")): ps.get("text", "")
                    for ps in parsed_steps
                }
                student_info = {
                    "subject":        task.subject     if task else "Unknown",
                    "name":           str(submission.student_id or "Student"),
                    "roll_number":    "—",
                    "class":          task.grade_level if task else "—",
                    "date":           datetime.now().strftime("%Y-%m-%d"),
                    "max_marks":      str(task.max_marks) if task else "—",
                    "obtained_marks": str(result_data["grade"]),
                }
                latex_src = _build_latex_transcript(
                    student_info,
                    rubric_steps,
                    answers_map,
                    result_data["step_grades"],
                    submission.raw_text or "",
                )
                from app.services.ingestion import upload_file as _upload
                latex_key = await asyncio.to_thread(
                    _upload,
                    latex_src.encode("utf-8"),
                    "transcript.tex",
                    "text/x-latex",
                    f"submissions/{submission.id}",
                )
                # Store key — use flag_modified so SQLAlchemy detects JSONB mutation
                new_pc = dict(submission.parsed_content)
                new_pc["latex_transcript_key"] = latex_key
                submission.parsed_content = new_pc
                attributes.flag_modified(submission, "parsed_content")
                logger.info(f"[P&G] LaTeX transcript stored: {latex_key}")
            except Exception as latex_err:
                logger.warning(f"[P&G] LaTeX transcript generation skipped: {latex_err}")

            # ── Persist GradeResult ───────────────────────────────────────────
            grade_result = GradeResult(
                id=uuid.uuid4(),
                submission_id=submission.id,
                grading_run_id=run.id,
                grade=result_data["grade"],
                max_grade=result_data["max_grade"],
                grade_distribution=result_data["grade_distribution"],
                confidence=result_data["confidence"],
                step_grades=result_data["step_grades"],
                justification=result_data["justification"],
                llm_call_ids=result_data.get("llm_call_ids", []),
                model_used=result_data["model_used"],
                latency_ms=result_data["latency_ms"],
                component_grades=result_data.get("component_grades"),
                review_status=result_data.get("review_status", "AUTO_GRADED"),
                review_reasons=result_data.get("review_reasons"),
                flagged_components=result_data.get("flagged_components"),
            )
            session.add(grade_result)

            if result_data.get("question_decomposition"):
                submission.question_decomposition = result_data["question_decomposition"]

            submission.status = "GRADED"
            run.status        = "COMPLETED"
            run.graded_count  = 1
            run.completed_at  = datetime.now(timezone.utc)
            await session.commit()

            logger.info(
                f"[P&G] {submission_id} → GRADED "
                f"{result_data['grade']}/{result_data['max_grade']} "
                f"(confidence={result_data['confidence']:.2f}, "
                f"latency={result_data['latency_ms']}ms)"
            )

        except Exception as e:
            logger.error(f"[P&G] Grade failed for {submission_id}: {e}", exc_info=True)
            try:
                result = await session.execute(
                    select(Submission).filter_by(id=sub_uuid)
                )
                sub = result.scalar_one_or_none()
                if sub:
                    sub.status        = "FAILED"
                    sub.error_message = f"Grading error: {str(e)}"
                    await session.commit()
            except Exception:
                await session.rollback()

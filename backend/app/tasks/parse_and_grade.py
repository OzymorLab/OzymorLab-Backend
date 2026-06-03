"""
parse_and_grade — end-to-end pipeline for a single answer sheet.

Upload → Parse → Grade, completely independent per submission.
"""
import asyncio
import io
import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


async def parse_and_grade(submission_id: str) -> None:
    """
    Full end-to-end pipeline for one answer sheet.
    Safe to call as a fire-and-forget asyncio task.
    """
    from app.db.session import async_session_factory
    from app.db.models import Submission, TaskRubric, GradingRun, GradeResult, Task, SubmissionStep
    from app.services.ingestion import download_file, upload_file
    from app.services.parsing import parse_submission as _parse_sync
    from app.services.grading import grade_submission as _grade_sync
    from app.config import settings
    from sqlalchemy import select, delete

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

            # Look up active approved rubric steps to help with Q&A alignment
            rubric_result = await session.execute(
                select(TaskRubric)
                .filter_by(task_id=task_id, is_active=True, approval_status="APPROVED")
                .order_by(TaskRubric.created_at.desc())
                .limit(1)
            )
            rubric = rubric_result.scalar_one_or_none()
            
            questions = None
            if rubric:
                questions = list(rubric.rubric_json.get("steps", []))

            # All sync/blocking work offloaded to thread pool
            file_data = await asyncio.to_thread(download_file, file_key)
            raw_text, parsed_content = await asyncio.to_thread(
                _parse_sync, file_data, file_type, str(submission.id), questions
            )

            # Store the full transcript as an intermediate file
            try:
                await asyncio.to_thread(
                    upload_file,
                    raw_text.encode("utf-8"),
                    "full_raw_transcript.txt",
                    "text/plain",
                    f"submissions/{submission.id}"
                )
            except Exception as e:
                logger.warning(f"[P&G] Failed to save raw transcript intermediate file: {e}")

            submission.raw_text      = raw_text
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
            # Re-fetch in a brand-new session to confirm PARSED is committed
            result = await session.execute(
                select(Submission).filter_by(id=sub_uuid)
            )
            submission = result.scalar_one_or_none()

            # Hard gate: only proceed if the DB confirms status is PARSED
            if not submission or submission.status != "PARSED":
                logger.info(
                    f"[P&G] {submission_id} not in PARSED state "
                    f"(got '{getattr(submission, 'status', 'none')}') — skipping auto-grade"
                )
                return

            # Find active, APPROVED rubric for this task
            rubric_result = await session.execute(
                select(TaskRubric)
                .filter_by(task_id=task_id, is_active=True, approval_status="APPROVED")
                .order_by(TaskRubric.created_at.desc())
                .limit(1)
            )
            rubric = rubric_result.scalar_one_or_none()

            if not rubric:
                logger.info(
                    f"[P&G] {submission_id} parsed — no approved rubric yet, "
                    "leaving as PARSED for manual grading"
                )
                return

            # Fetch task metadata for subject/board/grade_level
            task_result = await session.execute(
                select(Task).filter_by(id=task_id)
            )
            task = task_result.scalar_one_or_none()

            # Create a dedicated single-submission GradingRun
            run = GradingRun(
                id=uuid.uuid4(),
                task_id=task_id,
                rubric_version=rubric.version,
                model=settings.GEMINI_MODEL,
                temperature=settings.GRADING_TEMPERATURE,
                description=f"Auto-grade on upload",
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

            # Build rubric data dict exactly as bulk-grade does
            rubric_data = dict(rubric.rubric_json)
            rubric_data["grading_notes"] = rubric.grading_notes or ""
            rubric_data["model"]         = settings.GEMINI_MODEL

            # Run the grading pipeline in a thread (it's sync/LLM-bound)
            result_data = await asyncio.to_thread(
                _grade_sync,
                rubric_data,
                submission.parsed_content,
                settings.GRADING_TEMPERATURE,
                task.subject    if task else "General",
                task.board      if task else "Generic",
                task.grade_level if task else "Unknown",
                submission.file_key,
                str(submission.id),
                None,           # no BYOK key
            )

            # Save the generated LaTeX document as an intermediate file
            try:
                # We can generate LaTeX transcript with paired Q&A
                from tests.debug_pdf import build_latex, pdf_to_images
                # Try to import/build latex using details
                # Retrieve questions and answer text
                questions_list = list(rubric.rubric_json.get("steps", []))
                answers_map = {str(s.get("step_num")): s.get("text", "") for s in submission.parsed_content.get("steps", [])}
                
                # Transform rubric steps to expected questions format for build_latex
                latex_questions = [{"number": str(q.get("step_num")), "text": q.get("description")} for q in questions_list]
                
                # Minimal student info for build_latex
                student_info_placeholder = {
                    "subject": task.subject if task else "General",
                    "name": "Student",
                    "roll_number": "UNKNOWN",
                    "class": task.grade_level if task else "Unknown",
                    "section": "UNKNOWN",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "school": "UNKNOWN",
                    "max_marks": str(task.max_marks) if task else "100",
                    "obtained_marks": str(result_data["grade"])
                }
                
                latex_src = build_latex(student_info_placeholder, latex_questions, answers_map, raw_text)
                
                await asyncio.to_thread(
                    upload_file,
                    latex_src.encode("utf-8"),
                    "transcript.tex",
                    "application/x-latex",
                    f"submissions/{submission.id}"
                )
                
                # Store the key in parsed_content for API retrieval
                submission.parsed_content["latex_transcript_key"] = f"submissions/{submission.id}/transcript.tex"
            except Exception as e:
                logger.warning(f"[P&G] LaTeX transcript generation failed: {e}")

            # Persist GradeResult
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

            # Persist individual answer steps in the SubmissionStep database table
            await session.execute(
                delete(SubmissionStep).filter_by(submission_id=submission.id)
            )

            for sg in result_data["step_grades"]:
                s_num = sg["step_num"]
                p_step = next((s for s in submission.parsed_content.get("steps", []) if s.get("step_num") == s_num), None)
                
                step_text = ""
                step_latex = ""
                bbox_data = None
                
                if p_step:
                    step_text = p_step.get("text", "")
                    step_latex = ", ".join(p_step.get("equations", []))
                    diagrams = p_step.get("diagrams", [])
                    if diagrams:
                        bbox_data = {
                            "diagram_key": diagrams[0].get("key"),
                            "diagram_filename": diagrams[0].get("filename"),
                            "box": diagrams[0].get("box")
                        }

                sub_step = SubmissionStep(
                    submission_id=submission.id,
                    step_num=s_num,
                    step_type=sg.get("step_type") or (p_step.get("step_type") if p_step else "statement"),
                    text=step_text,
                    latex=step_latex,
                    marks_awarded=sg["marks_awarded"],
                    max_marks=sg["max_marks"],
                    justification=sg["justification"],
                    error_type=sg["error_type"],
                    bounding_box=bbox_data
                )
                session.add(sub_step)

            if result_data.get("question_decomposition"):
                submission.question_decomposition = result_data["question_decomposition"]

            submission.status   = "GRADED"
            run.status          = "COMPLETED"
            run.graded_count    = 1
            run.completed_at    = datetime.now(timezone.utc)
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

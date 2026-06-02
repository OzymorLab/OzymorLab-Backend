"""
parse_and_grade — end-to-end pipeline for a single answer sheet.

Upload → Parse → Grade, completely independent per submission.
No manual "start grading" button required.

Flow:
  1. Download file from S3 and parse it (OCR + step segmentation).
  2. Look up the task's active, APPROVED rubric.
  3. If one exists → create a single-submission GradingRun and grade immediately.
  4. If no approved rubric → leave submission as PARSED so teacher can grade later.

Every submission runs this pipeline in its own asyncio task, in its own DB
session. One submission crashing or being slow has zero effect on others.
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
    from app.db.models import Submission, TaskRubric, GradingRun, GradeResult, Task
    from app.services.ingestion import download_file
    from app.services.parsing import parse_submission as _parse_sync
    from app.services.grading import grade_submission as _grade_sync
    from app.config import settings
    from sqlalchemy import select

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

            # All sync/blocking work offloaded to thread pool
            file_data = await asyncio.to_thread(download_file, file_key)
            raw_text, parsed_content = await asyncio.to_thread(
                _parse_sync, file_data, file_type
            )

            # Diagram crop extraction (optional, non-fatal)
            diagram_file_key = await asyncio.to_thread(
                _extract_diagram_crop, file_data, file_type, submission_id
            )
            if diagram_file_key:
                parsed_content["diagram_file_key"] = diagram_file_key

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
            # Re-fetch submission (fresh session)
            result = await session.execute(
                select(Submission).filter_by(id=sub_uuid)
            )
            submission = result.scalar_one_or_none()
            if not submission or submission.status != "PARSED":
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
                created_by=submission.student_id,   # best available, may be None
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


def _extract_diagram_crop(
    file_data: bytes, file_type: str, submission_id: str
) -> str | None:
    """
    Sync helper: render the first PDF page as PNG and upload to S3.
    Returns the S3 key or None on failure. Runs in a thread pool.
    """
    if file_type.lower() != "pdf":
        return None
    try:
        import fitz
        from app.services.ingestion import upload_file_obj

        doc = fitz.open(stream=file_data, filetype="pdf")
        if len(doc) == 0:
            doc.close()
            return None
        page = doc[0]
        pix  = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_bytes = pix.tobytes("png")
        doc.close()

        key = upload_file_obj(io.BytesIO(img_bytes), f"diagram_crops/{submission_id}.png")
        return key
    except Exception as e:
        logger.warning(f"[P&G] Diagram crop failed for {submission_id}: {e}")
        return None

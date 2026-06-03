"""
Background task: parse a submission.
Downloads file from S3 → extracts text → segments steps → updates DB.
"""
import asyncio
import io
import logging
import uuid

from app.db.session import async_session_factory

logger = logging.getLogger(__name__)


async def parse(submission_id: str):
    """
    Parse a submission: download from S3, extract text, segment steps.
    """
    from app.db.models import Submission, TaskRubric
    from sqlalchemy import select

    async with async_session_factory() as session:
        try:
            result = await session.execute(
                select(Submission).filter_by(id=uuid.UUID(submission_id))
            )
            submission = result.scalar_one_or_none()
            if not submission:
                logger.warning(f"[Parse] Submission {submission_id} not found")
                return

            file_key = submission.file_key
            file_type = submission.file_type or "pdf"
            task_id = submission.task_id

            # Mark as PARSING immediately
            submission.status = "PARSING"
            await session.commit()
            logger.info(f"[Parse] {submission_id} → PARSING")

            # ── 1. Download from S3 (network I/O, sync SDK) ──────────────────
            from app.services.ingestion import download_file
            file_data: bytes = await asyncio.to_thread(download_file, file_key)

            # Query active approved rubric steps to help with Q&A alignment
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

            # ── 2. Parse PDF/image (CPU-bound) ────────────────────────────────
            from app.services.parsing import parse_submission as _parse_sync
            raw_text, parsed_content = await asyncio.to_thread(
                _parse_sync, file_data, file_type, str(submission.id), questions
            )

            # Upload the raw transcript as an intermediate file
            from app.services.ingestion import upload_file
            try:
                await asyncio.to_thread(
                    upload_file,
                    raw_text.encode("utf-8"),
                    "full_raw_transcript.txt",
                    "text/plain",
                    f"submissions/{submission.id}"
                )
            except Exception as e:
                logger.warning(f"[Parse] Failed to save raw transcript intermediate file: {e}")

            # ── 4. Persist results ────────────────────────────────────────────
            submission.raw_text = raw_text
            submission.parsed_content = parsed_content
            submission.status = "PARSED"
            submission.error_message = None
            await session.commit()

            logger.info(
                f"[Parse] {submission_id} → PARSED "
                f"({len(parsed_content.get('steps', []))} steps, "
                f"confidence={parsed_content.get('parse_confidence', 0):.2f})"
            )

        except Exception as e:
            logger.error(f"[Parse] Failed for {submission_id}: {e}", exc_info=True)
            try:
                result = await session.execute(
                    select(Submission).filter_by(id=uuid.UUID(submission_id))
                )
                submission = result.scalar_one_or_none()
                if submission:
                    submission.status = "FAILED"
                    submission.error_message = str(e)
                    await session.commit()
            except Exception:
                await session.rollback()

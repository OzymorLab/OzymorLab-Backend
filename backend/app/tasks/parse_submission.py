"""
Background task: parse a submission.
Downloads file from S3 → extracts text → segments steps → updates DB.

All sync/CPU-bound work (S3 download, PDF parsing, diagram crop upload) runs
in a thread pool via asyncio.to_thread so it never blocks the event loop.
This means multiple submissions can be parsed truly in parallel — one stuck
upload cannot delay another.
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
    Updates the submission record in the database.
    Each step that is sync/CPU-bound is offloaded to a thread pool so the
    async event loop is never blocked.
    """
    from app.db.models import Submission
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

            # Mark as PARSING immediately
            submission.status = "PARSING"
            await session.commit()
            logger.info(f"[Parse] {submission_id} → PARSING")

            # ── 1. Download from S3 (network I/O, sync SDK) ──────────────────
            from app.services.ingestion import download_file
            file_data: bytes = await asyncio.to_thread(download_file, file_key)

            # ── 2. Parse PDF/image (CPU-bound) ────────────────────────────────
            from app.services.parsing import parse_submission as _parse_sync
            raw_text, parsed_content = await asyncio.to_thread(
                _parse_sync, file_data, file_type
            )

            # ── 3. Diagram crop extraction (CPU-bound + S3 upload) ────────────
            diagram_file_key = None
            if parsed_content.get("has_diagrams") and file_type.lower() == "pdf":
                diagram_file_key = await asyncio.to_thread(
                    _extract_and_upload_diagram_crop,
                    file_data,
                    submission_id,
                )

            if diagram_file_key:
                parsed_content["diagram_file_key"] = diagram_file_key

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


def _extract_and_upload_diagram_crop(file_data: bytes, submission_id: str) -> str | None:
    """
    Sync helper: render the first PDF page as PNG and upload to S3.
    Runs inside a thread pool — safe to use sync libs here.
    Returns the S3 key of the uploaded crop, or None on failure.
    """
    try:
        import fitz
        from app.services.ingestion import upload_file_obj

        doc = fitz.open(stream=file_data, filetype="pdf")
        if len(doc) == 0:
            doc.close()
            return None

        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_bytes = pix.tobytes("png")
        doc.close()

        img_file = io.BytesIO(img_bytes)
        key = upload_file_obj(img_file, f"diagram_crops/{submission_id}.png")
        logger.info(f"[Parse] Diagram crop uploaded: {key}")
        return key
    except Exception as e:
        logger.error(f"[Parse] Diagram crop extraction failed for {submission_id}: {e}")
        return None

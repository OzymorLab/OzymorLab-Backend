"""
Background task: parse a submission.
Downloads file from S3 → extracts text → segments steps → updates DB.
Plain async function — no Celery required.
"""
import logging
import uuid

from app.db.session import async_session_factory
from app.services.ingestion import download_file
from app.services.parsing import parse_submission

logger = logging.getLogger(__name__)


async def parse(submission_id: str):
    """
    Parse a submission: download from S3, extract text, segment steps.
    Updates the submission record in the database.
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
                logger.warning(f"Submission {submission_id} not found")
                return

            # Update status to PARSING
            submission.status = "PARSING"
            await session.commit()

            # Download file from S3 (sync call — runs in the current thread)
            file_data = download_file(submission.file_key)
            file_type = submission.file_type or "pdf"

            # Run the 3-pass parsing pipeline (sync, CPU-bound)
            raw_text, parsed_content = parse_submission(file_data, file_type)

            # Diagram image extraction for DEIS
            diagram_file_key = None
            if parsed_content.get("has_diagrams") and file_type.lower() == "pdf":
                try:
                    import fitz
                    import io
                    from app.services.ingestion import upload_file_obj

                    doc = fitz.open(stream=file_data, filetype="pdf")
                    if len(doc) > 0:
                        page = doc[0]
                        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                        img_bytes = pix.tobytes("png")

                        img_file = io.BytesIO(img_bytes)
                        diagram_file_key = upload_file_obj(
                            img_file, f"diagram_crops/{submission_id}.png"
                        )
                        logger.info(
                            f"Extracted diagram image for DEIS: {diagram_file_key}"
                        )
                    doc.close()
                except Exception as e:
                    logger.error(
                        f"Failed to extract diagram image from PDF: {e}"
                    )

            if diagram_file_key:
                parsed_content["diagram_file_key"] = diagram_file_key

            # Persist results
            submission.raw_text = raw_text
            submission.parsed_content = parsed_content
            submission.status = "PARSED"
            submission.error_message = None
            await session.commit()

            logger.info(
                f"Submission {submission_id} parsed successfully: "
                f"{len(parsed_content.get('steps', []))} steps, "
                f"confidence={parsed_content.get('parse_confidence', 0):.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to parse submission {submission_id}: {e}")
            try:
                from sqlalchemy import select
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

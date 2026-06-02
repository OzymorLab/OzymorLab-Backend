"""
Background task: extract student identity from the first page of a submission PDF.
Plain async function — no Celery required.
"""
import logging
import json
import uuid

from pydantic import BaseModel, Field

from app.db.session import async_session_factory
from app.services.ingestion import download_file
from app.services.llm_client import get_client as get_gemini_client

logger = logging.getLogger(__name__)


class StudentIdentityExtraction(BaseModel):
    name: str = Field(description="The student's handwritten name on the paper")
    roll_number: str = Field(description="The student's handwritten roll number on the paper")
    class_name: str = Field(description="The student's class, e.g., 'X', 'XII', '10'")
    section: str = Field(description="The student's section, e.g., 'A', 'B', 'Science'")
    subject: str = Field(description="The subject of the exam")


async def extract(submission_id: str):
    """
    Extract student identity from a submission's first page and update the DB.
    On failure, transitions the submission to IDENTITY_EXTRACTED anyway so the
    pipeline can continue with parse → grade.
    """
    from app.db.models import Submission, Student
    from sqlalchemy import select

    async with async_session_factory() as session:
        try:
            result = await session.execute(
                select(Submission).filter_by(id=uuid.UUID(submission_id))
            )
            submission = result.scalar_one_or_none()
            if not submission:
                logger.error(f"Submission {submission_id} not found")
                return

            # Download file (sync)
            file_data = download_file(submission.file_key)

            # Render first page as PNG
            file_type = submission.file_type or "pdf"
            if file_type.lower() == "pdf":
                import fitz

                doc = fitz.open(stream=file_data, filetype="pdf")
                if len(doc) == 0:
                    raise ValueError("Empty PDF")
                page = doc[0]
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                img_bytes = pix.tobytes("png")
                doc.close()
            else:
                img_bytes = file_data

            # Ask Gemini to extract identity
            client = get_gemini_client()
            prompt = "Extract the student's identity details from the top of this answer sheet."

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    {"mime_type": "image/png", "data": img_bytes},
                    prompt,
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": StudentIdentityExtraction,
                    "temperature": 0.1,
                },
            )

            identity = json.loads(response.text)
            logger.info(f"Extracted identity for {submission_id}: {identity}")

            # Map to an existing student by roll number
            student_result = await session.execute(
                select(Student).filter_by(
                    roll_number=identity.get("roll_number")
                )
            )
            student = student_result.scalar_one_or_none()
            if student:
                submission.student_id = student.id
                logger.info(
                    f"Mapped submission {submission_id} to student {student.id}"
                )
            else:
                logger.warning(
                    f"Could not find student with roll number "
                    f"{identity.get('roll_number')}"
                )

            submission.status = "IDENTITY_EXTRACTED"
            await session.commit()

        except Exception as e:
            logger.error(
                f"Failed to extract identity for {submission_id}: {e}"
            )
            # Transition to IDENTITY_EXTRACTED anyway so the pipeline can continue
            try:
                from sqlalchemy import select
                result = await session.execute(
                    select(Submission).filter_by(id=uuid.UUID(submission_id))
                )
                submission = result.scalar_one_or_none()
                if submission:
                    submission.status = "IDENTITY_EXTRACTED"
                    await session.commit()
            except Exception:
                await session.rollback()

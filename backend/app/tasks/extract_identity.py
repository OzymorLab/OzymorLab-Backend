"""
Celery task: extract student identity from the first page of a submission PDF.
"""
import logging
import fitz
import json
from pydantic import BaseModel, Field
from uuid import UUID

from app.worker import celery_app
from app.db.session import get_sync_session
from app.db.models import Submission, Student
from app.services.ingestion import download_file
from app.services.llm_client import get_client as get_gemini_client

logger = logging.getLogger(__name__)

class StudentIdentityExtraction(BaseModel):
    name: str = Field(description="The student's handwritten name on the paper")
    roll_number: str = Field(description="The student's handwritten roll number on the paper")
    class_name: str = Field(description="The student's class, e.g., 'X', 'XII', '10'")
    section: str = Field(description="The student's section, e.g., 'A', 'B', 'Science'")
    subject: str = Field(description="The subject of the exam")

@celery_app.task(bind=True, name="app.tasks.extract_identity.extract", max_retries=2)
def extract(self, submission_id: str):
    session = get_sync_session()
    try:
        submission = session.query(Submission).filter_by(id=submission_id).first()
        if not submission:
            logger.error(f"Submission {submission_id} not found")
            return {"error": "Submission not found"}

        # Download file
        file_data = download_file(submission.file_key)
        
        # Render first page
        file_type = submission.file_type or "pdf"
        if file_type.lower() == "pdf":
            doc = fitz.open(stream=file_data, filetype="pdf")
            if len(doc) == 0:
                raise ValueError("Empty PDF")
                
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            img_bytes = pix.tobytes("png")
            doc.close()
        else:
            # If it's already an image
            img_bytes = file_data
        
        # Prompt Gemini to extract identity
        client = get_gemini_client()
        prompt = "Extract the student's identity details from the top of this answer sheet."
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                {"mime_type": "image/png", "data": img_bytes},
                prompt
            ],
            config={
                'response_mime_type': 'application/json',
                'response_schema': StudentIdentityExtraction,
                'temperature': 0.1
            }
        )
        
        identity = json.loads(response.text)
        logger.info(f"Extracted identity: {identity}")
        
        # Map to an existing student by roll number
        # Note: In a real multi-tenant app, we would also scope by school_id/section_id
        student = session.query(Student).filter_by(roll_number=identity.get("roll_number")).first()
        if student:
            submission.student_id = student.id
            logger.info(f"Mapped submission {submission_id} to student {student.id}")
        else:
            logger.warning(f"Could not find student with roll number {identity.get('roll_number')}")
            
        submission.status = "IDENTITY_EXTRACTED"
        session.commit()
        return {"status": "success", "identity": identity, "mapped_student_id": str(student.id) if student else None}
        
    except Exception as e:
        logger.error(f"Failed to extract identity for {submission_id}: {e}")
        session.rollback()
        # Transition directly to PARSING if identity fails, so we don't break the pipeline
        try:
            submission = session.query(Submission).filter_by(id=submission_id).first()
            if submission:
                submission.status = "IDENTITY_EXTRACTED" # Still transition to allow orchestrator to proceed
                session.commit()
        except:
            pass
        raise self.retry(exc=e, countdown=2 ** self.request.retries)
    finally:
        session.close()

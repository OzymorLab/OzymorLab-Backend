"""
Celery task: parse a submission.
Downloads file from S3 → extracts text → segments steps → updates DB.
"""
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.worker import celery_app
from app.config import settings
from app.services.ingestion import download_file
from app.services.parsing import parse_submission

logger = logging.getLogger(__name__)


from app.db.session import get_sync_session


@celery_app.task(bind=True, name="app.tasks.parse_submission.parse", max_retries=2)
def parse(self, submission_id: str):
    """
    Parse a submission: download from S3, extract text, segment steps.
    Updates the submission record in PostgreSQL.
    """
    from app.db.models import Submission

    session = get_sync_session()
    try:
        # Load submission
        submission = session.query(Submission).filter_by(id=submission_id).first()
        if not submission:
            logger.error(f"Submission {submission_id} not found")
            return {"error": "Submission not found"}

        # Update status to PARSING
        submission.status = "PARSING"
        session.commit()

        # Download file from S3
        file_data = download_file(submission.file_key)
        file_type = submission.file_type or "pdf"

        # Run the 3-pass parsing pipeline
        raw_text, parsed_content = parse_submission(file_data, file_type)

        # Update submission with parsed results
        submission.raw_text = raw_text
        submission.parsed_content = parsed_content
        submission.status = "PARSED"
        submission.error_message = None
        session.commit()

        logger.info(
            f"Submission {submission_id} parsed successfully: "
            f"{len(parsed_content.get('steps', []))} steps, "
            f"confidence={parsed_content.get('parse_confidence', 0):.2f}"
        )

        return {
            "submission_id": submission_id,
            "status": "PARSED",
            "steps_count": len(parsed_content.get("steps", [])),
            "confidence": parsed_content.get("parse_confidence", 0),
        }

    except Exception as e:
        logger.error(f"Failed to parse submission {submission_id}: {e}")
        try:
            submission = session.query(Submission).filter_by(id=submission_id).first()
            if submission:
                submission.status = "FAILED"
                submission.error_message = str(e)
                session.commit()
        except Exception:
            session.rollback()

        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=2 ** self.request.retries)

    finally:
        session.close()

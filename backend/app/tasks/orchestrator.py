"""
Celery task: Orchestrate the auto-grading pipeline.
Workflow: UPLOADED → (extract_identity) → IDENTITY_EXTRACTED → (parse_submission) → PARSED → (grade_submission) → GRADED
"""
import logging
from celery import chain

from app.worker import celery_app
from app.tasks.extract_identity import extract as extract_task
from app.tasks.parse_submission import parse as parse_task
from app.tasks.grade_submission import grade as grade_task
from app.db.session import get_sync_session
from app.db.models import Submission, GradingRun

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name="app.tasks.orchestrator.process_submission")
def process_submission(self, submission_id: str, grading_run_id: str = None):
    """
    Kicks off the sequential pipeline for a single submission.
    
    If grading_run_id is not provided, this implies we want to grade it immediately using 
    the active rubric for the task. We'll need to create a GradingRun on the fly, or 
    the pipeline can just run without a run_id (but grade_submission currently requires a run_id).
    For now, assume the caller creates a GradingRun (or we create one here).
    """
    logger.info(f"Starting orchestration for submission {submission_id}")
    
    session = get_sync_session()
    try:
        submission = session.query(Submission).filter_by(id=submission_id).first()
        if not submission:
            logger.error(f"Submission {submission_id} not found")
            return
            
        # Create a default GradingRun if not provided, so the grade task has context
        if not grading_run_id:
            from app.db.models import TaskRubric
            # Find the active rubric for this task
            rubric = session.query(TaskRubric).filter_by(task_id=submission.task_id, is_active=True).order_by(TaskRubric.created_at.desc()).first()
            
            if not rubric:
                logger.error(f"No active rubric found for task {submission.task_id}. Cannot auto-grade.")
                # We can still extract and parse
                pipeline = chain(
                    extract_task.s(submission_id),
                    parse_task.s(submission_id)
                )
                pipeline.apply_async()
                return
                
            from app.config import settings
            run = GradingRun(
                task_id=submission.task_id,
                rubric_version=rubric.version,
                model=settings.GEMINI_MODEL,
                temperature=0.0,
                description="Auto-triggered grading run",
                status="RUNNING",
                total_submissions=1
            )
            session.add(run)
            session.commit()
            grading_run_id = str(run.id)

        # Trigger the celery chain
        # 1. Extract Identity -> 2. Parse PDF -> 3. Grade
        
        pipeline = chain(
            extract_task.si(submission_id),
            parse_task.si(submission_id),
            grade_task.si(submission_id, grading_run_id)
        )
        
        pipeline.apply_async()
        
    except Exception as e:
        logger.error(f"Failed to orchestrate submission {submission_id}: {e}")
    finally:
        session.close()

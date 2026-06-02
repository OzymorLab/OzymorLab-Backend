"""
Orchestrate the auto-grading pipeline for a single submission.
Workflow: UPLOADED → extract_identity → IDENTITY_EXTRACTED → parse → PARSED → grade → GRADED
Plain async function — no Celery required.
"""
import logging
import uuid

from app.db.session import async_session_factory
from app.tasks.extract_identity import extract as extract_task
from app.tasks.parse_submission import parse as parse_task
from app.tasks.grade_submission import grade as grade_task

logger = logging.getLogger(__name__)


async def process_submission(submission_id: str, grading_run_id: str = None):
    """
    Run the sequential pipeline for a single submission:
      1. Extract student identity
      2. Parse PDF content
      3. Grade against the active rubric

    If grading_run_id is not provided, a GradingRun is created automatically
    using the task's active rubric. If no rubric exists, only extract + parse
    are performed.
    """
    from app.db.models import Submission, GradingRun, TaskRubric
    from app.config import settings
    from sqlalchemy import select

    logger.info(f"Starting orchestration for submission {submission_id}")

    async with async_session_factory() as session:
        result = await session.execute(
            select(Submission).filter_by(id=uuid.UUID(submission_id))
        )
        submission = result.scalar_one_or_none()
        if not submission:
            logger.error(f"Submission {submission_id} not found")
            return

        if not grading_run_id:
            # Try to find an active rubric and create a run on the fly
            rubric_result = await session.execute(
                select(TaskRubric)
                .filter_by(task_id=submission.task_id, is_active=True)
                .order_by(TaskRubric.created_at.desc())
                .limit(1)
            )
            rubric = rubric_result.scalar_one_or_none()

            if rubric:
                run = GradingRun(
                    task_id=submission.task_id,
                    rubric_version=rubric.version,
                    model=settings.GEMINI_MODEL,
                    temperature=0.0,
                    description="Auto-triggered grading run",
                    status="RUNNING",
                    total_submissions=1,
                )
                session.add(run)
                await session.commit()
                await session.refresh(run)
                grading_run_id = str(run.id)
            else:
                logger.warning(
                    f"No active rubric for task {submission.task_id}. "
                    "Running extract + parse only."
                )

    try:
        # Step 1: Extract identity
        await extract_task(submission_id)

        # Step 2: Parse PDF
        await parse_task(submission_id)

        # Step 3: Grade (only if we have a run)
        if grading_run_id:
            await grade_task(submission_id, grading_run_id)

    except Exception as e:
        logger.error(
            f"Orchestration failed for submission {submission_id}: {e}",
            exc_info=True,
        )

"""
Celery task: grade a submission.
Loads rubric → runs hybrid grading pipeline → stores results → triggers drift check.
"""
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.worker import celery_app
from app.config import settings
from app.services.grading import grade_submission

logger = logging.getLogger(__name__)


from app.db.session import get_sync_session


@celery_app.task(bind=True, name="app.tasks.grade_submission.grade", max_retries=1)
def grade(self, submission_id: str, grading_run_id: str):
    """
    Grade a single submission as part of a grading run.
    """
    from app.db.models import Submission, GradingRun, GradeResult, TaskRubric, Task

    session = get_sync_session()
    try:
        # Load submission, run, and task
        submission = session.query(Submission).filter_by(id=submission_id).first()
        run = session.query(GradingRun).filter_by(id=grading_run_id).first()

        if not submission or not run:
            logger.error(f"Submission {submission_id} or run {grading_run_id} not found")
            return {"error": "Not found"}
            
        task = session.query(Task).filter_by(id=run.task_id).first()

        if submission.status != "PARSED" or not submission.parsed_content:
            logger.warning(f"Submission {submission_id} not in PARSED state (current: {submission.status})")
            run.failed_count = (run.failed_count or 0) + 1
            session.commit()
            return {"error": "Submission not parsed"}

        # Load rubric
        rubric_record = session.query(TaskRubric).filter_by(
            task_id=run.task_id, version=run.rubric_version
        ).first()

        if not rubric_record:
            logger.error(f"Rubric version {run.rubric_version} not found for task {run.task_id}")
            run.failed_count = (run.failed_count or 0) + 1
            session.commit()
            return {"error": "Rubric not found"}

        # Update submission status
        submission.status = "GRADING"
        session.commit()

        # Run the hybrid grading pipeline
        rubric_data = rubric_record.rubric_json
        rubric_data["grading_notes"] = rubric_record.grading_notes or ""
        rubric_data["model"] = run.model

        # Load BYOK Gemini key if the run creator has one
        from app.db.models import User
        user_gemini_key = None
        if run.created_by:
            user = session.query(User).filter_by(id=run.created_by).first()
            if user and user.gemini_api_key:
                user_gemini_key = user.gemini_api_key

        result = grade_submission(
            rubric=rubric_data,
            parsed_content=submission.parsed_content,
            temperature=run.temperature,
            subject=task.subject if task else "General",
            board=task.board if task else "Generic",
            grade_level=task.grade_level if task else "Unknown",
            file_key=submission.file_key,
            submission_id=str(submission.id),
            user_gemini_key=user_gemini_key,
        )

        # Store grade result
        grade_result = GradeResult(
            id=uuid.uuid4(),
            submission_id=submission.id,
            grading_run_id=run.id,
            grade=result["grade"],
            max_grade=result["max_grade"],
            grade_distribution=result["grade_distribution"],
            confidence=result["confidence"],
            step_grades=result["step_grades"],
            justification=result["justification"],
            llm_call_ids=result.get("llm_call_ids", []),
            model_used=result["model_used"],
            latency_ms=result["latency_ms"],
            # Multimodal evaluation fields
            component_grades=result.get("component_grades"),
            review_status=result.get("review_status", "AUTO_GRADED"),
            review_reasons=result.get("review_reasons"),
            flagged_components=result.get("flagged_components"),
        )
        session.add(grade_result)

        # Cache question decomposition on the submission
        if result.get("question_decomposition"):
            submission.question_decomposition = result["question_decomposition"]

        # Update submission and run counters
        submission.status = "GRADED"
        run.graded_count = (run.graded_count or 0) + 1
        session.commit()

        logger.info(
            f"Submission {submission_id} graded: {result['grade']}/{result['max_grade']} "
            f"(confidence={result['confidence']:.2f}, latency={result['latency_ms']}ms)"
        )

        return {
            "submission_id": submission_id,
            "grade": result["grade"],
            "max_grade": result["max_grade"],
            "confidence": result["confidence"],
        }

    except Exception as e:
        logger.error(f"Failed to grade submission {submission_id}: {e}")
        try:
            submission = session.query(Submission).filter_by(id=submission_id).first()
            run = session.query(GradingRun).filter_by(id=grading_run_id).first()
            if submission:
                submission.status = "FAILED"
                submission.error_message = str(e)
            if run:
                run.failed_count = (run.failed_count or 0) + 1
            session.commit()
        except Exception:
            session.rollback()

        raise self.retry(exc=e, countdown=5)

    finally:
        session.close()


@celery_app.task(name="app.tasks.grade_submission.finalize_run")
def finalize_run(grading_run_id: str):
    """
    Called after all submissions in a run are graded.
    Marks run as completed and triggers drift detection.
    """
    from app.db.models import GradingRun, Task

    session = get_sync_session()
    try:
        run = session.query(GradingRun).filter_by(id=grading_run_id).first()
        if not run:
            return

        run.status = "COMPLETED"
        run.completed_at = datetime.now(timezone.utc)
        session.commit()

        # Auto-trigger drift detection if a baseline exists
        task = session.query(Task).filter_by(id=run.task_id).first()
        if task and task.baseline_run_id and str(task.baseline_run_id) != str(run.id):
            from app.tasks.grade_submission import run_drift_detection
            run_drift_detection.delay(str(run.id), str(task.baseline_run_id), str(task.id))

        logger.info(f"Grading run {grading_run_id} finalized: {run.graded_count} graded, {run.failed_count} failed")

    finally:
        session.close()


@celery_app.task(name="app.tasks.grade_submission.run_drift_detection")
def run_drift_detection(current_run_id: str, baseline_run_id: str, task_id: str):
    """Run drift detection comparing current run to baseline."""
    from app.db.models import GradeResult, DriftReport, GradingAlert, Task
    from app.services.drift_detector import detect_drift, generate_alerts, compute_run_statistics

    session = get_sync_session()
    try:
        current_results = session.query(GradeResult).filter_by(grading_run_id=current_run_id).all()
        baseline_results = session.query(GradeResult).filter_by(grading_run_id=baseline_run_id).all()

        task = session.query(Task).filter_by(id=task_id).first()
        max_grade = task.max_marks if task else 20

        current_dicts = [{"grade_distribution": r.grade_distribution, "grade": r.grade,
                          "confidence": r.confidence, "latency_ms": r.latency_ms,
                          "step_grades": r.step_grades} for r in current_results]
        baseline_dicts = [{"grade_distribution": r.grade_distribution, "grade": r.grade,
                          "confidence": r.confidence, "latency_ms": r.latency_ms,
                          "step_grades": r.step_grades} for r in baseline_results]

        drift = detect_drift(current_dicts, baseline_dicts, max_grade)
        run_stats = compute_run_statistics(current_dicts, max_grade)

        # Store drift report
        report = DriftReport(
            task_id=task_id,
            current_run_id=current_run_id,
            baseline_run_id=baseline_run_id,
            kl_divergence=drift["kl_divergence"],
            mean_shift=drift["mean_shift"],
            entropy_current=drift["entropy_current"],
            entropy_baseline=drift["entropy_baseline"],
            drift_detected=drift["drift_detected"],
            severity=drift["severity"],
            details=drift["details"],
        )
        session.add(report)

        # Generate and store alerts
        alerts = generate_alerts(drift, run_stats, current_run_id)
        for alert_data in alerts:
            alert = GradingAlert(
                run_id=current_run_id,
                task_id=task_id,
                **alert_data,
            )
            session.add(alert)

        session.commit()
        logger.info(f"Drift detection complete for run {current_run_id}: severity={drift['severity']}")

    finally:
        session.close()

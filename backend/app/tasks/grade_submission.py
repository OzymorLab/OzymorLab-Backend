"""
Background tasks: grade a submission, finalize a run, run drift detection.
All tasks are plain async functions — no Celery required.
"""
import logging
import uuid
from datetime import datetime, timezone

from app.db.session import async_session_factory
from app.config import settings
from app.services.grading import grade_submission

logger = logging.getLogger(__name__)


async def grade(submission_id: str, grading_run_id: str):
    """
    Grade a single submission as part of a grading run.
    Updates the DB with the GradeResult and bumps run counters.
    """
    from app.db.models import Submission, GradingRun, GradeResult, TaskRubric, Task

    async with async_session_factory() as session:
        try:
            from sqlalchemy import select

            sub_result = await session.execute(
                select(Submission).filter_by(id=uuid.UUID(submission_id))
            )
            submission = sub_result.scalar_one_or_none()

            run_result = await session.execute(
                select(GradingRun).filter_by(id=uuid.UUID(grading_run_id))
            )
            run = run_result.scalar_one_or_none()

            if not submission or not run:
                logger.warning(
                    f"Submission {submission_id} or run {grading_run_id} not found"
                )
                return

            task_result = await session.execute(
                select(Task).filter_by(id=run.task_id)
            )
            task = task_result.scalar_one_or_none()

            if submission.status != "PARSED" or not submission.parsed_content:
                logger.warning(
                    f"Submission {submission_id} not in PARSED state "
                    f"(current: {submission.status})"
                )
                run.failed_count = (run.failed_count or 0) + 1
                await session.commit()
                return

            # Load rubric
            rubric_result = await session.execute(
                select(TaskRubric).filter_by(
                    task_id=run.task_id, version=run.rubric_version
                )
            )
            rubric_record = rubric_result.scalar_one_or_none()

            if not rubric_record:
                logger.error(
                    f"Rubric version {run.rubric_version} not found for task {run.task_id}"
                )
                run.failed_count = (run.failed_count or 0) + 1
                await session.commit()
                return

            # Update submission status
            submission.status = "GRADING"
            await session.commit()

            # Run the hybrid grading pipeline (sync, CPU-bound)
            rubric_data = rubric_record.rubric_json
            rubric_data["grading_notes"] = rubric_record.grading_notes or ""
            rubric_data["model"] = run.model

            result = grade_submission(
                rubric=rubric_data,
                parsed_content=submission.parsed_content,
                temperature=run.temperature,
                subject=task.subject if task else "General",
                board=task.board if task else "Generic",
                grade_level=task.grade_level if task else "Unknown",
                file_key=submission.file_key,
                submission_id=str(submission.id),
                user_gemini_key=None,
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
                component_grades=result.get("component_grades"),
                review_status=result.get("review_status", "AUTO_GRADED"),
                review_reasons=result.get("review_reasons"),
                flagged_components=result.get("flagged_components"),
            )
            session.add(grade_result)

            if result.get("question_decomposition"):
                submission.question_decomposition = result["question_decomposition"]

            submission.status = "GRADED"
            run.graded_count = (run.graded_count or 0) + 1
            await session.commit()

            logger.info(
                f"Submission {submission_id} graded: "
                f"{result['grade']}/{result['max_grade']} "
                f"(confidence={result['confidence']:.2f}, "
                f"latency={result['latency_ms']}ms)"
            )

        except Exception as e:
            logger.error(f"Failed to grade submission {submission_id}: {e}")
            try:
                from sqlalchemy import select
                sub_result = await session.execute(
                    select(Submission).filter_by(id=uuid.UUID(submission_id))
                )
                submission = sub_result.scalar_one_or_none()
                run_result = await session.execute(
                    select(GradingRun).filter_by(id=uuid.UUID(grading_run_id))
                )
                run = run_result.scalar_one_or_none()
                if submission:
                    submission.status = "FAILED"
                    submission.error_message = str(e)
                if run:
                    run.failed_count = (run.failed_count or 0) + 1
                await session.commit()
            except Exception:
                await session.rollback()


async def finalize_run(grading_run_id: str):
    """
    Called after all submissions in a run are graded.
    Marks run as COMPLETED and optionally triggers drift detection.
    """
    from app.db.models import GradingRun, Task
    from sqlalchemy import select

    async with async_session_factory() as session:
        try:
            run_result = await session.execute(
                select(GradingRun).filter_by(id=uuid.UUID(grading_run_id))
            )
            run = run_result.scalar_one_or_none()
            if not run:
                return

            run.status = "COMPLETED"
            run.completed_at = datetime.now(timezone.utc)
            await session.commit()

            # Auto-trigger drift detection if a baseline exists
            task_result = await session.execute(
                select(Task).filter_by(id=run.task_id)
            )
            task = task_result.scalar_one_or_none()
            if task and task.baseline_run_id and str(task.baseline_run_id) != str(run.id):
                import asyncio
                asyncio.create_task(
                    run_drift_detection(
                        str(run.id),
                        str(task.baseline_run_id),
                        str(task.id),
                    )
                )

            logger.info(
                f"Grading run {grading_run_id} finalized: "
                f"{run.graded_count} graded, {run.failed_count} failed"
            )

        except Exception as e:
            logger.error(f"Failed to finalize run {grading_run_id}: {e}")


async def run_drift_detection(
    current_run_id: str, baseline_run_id: str, task_id: str
):
    """Run drift detection comparing current run to baseline."""
    from app.db.models import GradeResult, DriftReport, GradingAlert, Task
    from app.services.drift_detector import detect_drift, generate_alerts, compute_run_statistics
    from sqlalchemy import select

    async with async_session_factory() as session:
        try:
            current_res = await session.execute(
                select(GradeResult).filter_by(grading_run_id=uuid.UUID(current_run_id))
            )
            current_results = current_res.scalars().all()

            baseline_res = await session.execute(
                select(GradeResult).filter_by(grading_run_id=uuid.UUID(baseline_run_id))
            )
            baseline_results = baseline_res.scalars().all()

            task_res = await session.execute(
                select(Task).filter_by(id=uuid.UUID(task_id))
            )
            task = task_res.scalar_one_or_none()
            max_grade = task.max_marks if task else 20

            def _to_dict(r):
                return {
                    "grade_distribution": r.grade_distribution,
                    "grade": r.grade,
                    "confidence": r.confidence,
                    "latency_ms": r.latency_ms,
                    "step_grades": r.step_grades,
                }

            current_dicts = [_to_dict(r) for r in current_results]
            baseline_dicts = [_to_dict(r) for r in baseline_results]

            drift = detect_drift(current_dicts, baseline_dicts, max_grade)
            run_stats = compute_run_statistics(current_dicts, max_grade)

            report = DriftReport(
                task_id=uuid.UUID(task_id),
                current_run_id=uuid.UUID(current_run_id),
                baseline_run_id=uuid.UUID(baseline_run_id),
                kl_divergence=drift["kl_divergence"],
                mean_shift=drift["mean_shift"],
                entropy_current=drift["entropy_current"],
                entropy_baseline=drift["entropy_baseline"],
                drift_detected=drift["drift_detected"],
                severity=drift["severity"],
                details=drift["details"],
            )
            session.add(report)

            alerts = generate_alerts(drift, run_stats, current_run_id)
            for alert_data in alerts:
                alert = GradingAlert(
                    run_id=uuid.UUID(current_run_id),
                    task_id=uuid.UUID(task_id),
                    **alert_data,
                )
                session.add(alert)

            await session.commit()
            logger.info(
                f"Drift detection complete for run {current_run_id}: "
                f"severity={drift['severity']}"
            )

        except Exception as e:
            logger.error(f"Drift detection failed for run {current_run_id}: {e}")

"""
Runs API — grading run management, batch grading, statistics, drift.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import GradingRun, GradeResult, Submission, TaskRubric, Task, DriftReport, User
from app.schemas.common import ApiResponse
from app.schemas.grade import GradingRunCreate, GradingRunResponse, RunStatistics, StepGradeResult, GradeResultResponse
from app.schemas.observability import DriftReportResponse
from app.config import settings
from app.services.auth_service import (
    get_current_user,
    require_role,
    check_task_access,
    check_run_access
)
import uuid

router = APIRouter(
    prefix="/runs", 
    tags=["Grading Runs"],
    dependencies=[Depends(require_role(["teacher", "admin", "hod", "principal"]))]
)


@router.post("")
async def create_run(payload: GradingRunCreate, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Create a new grading run configuration."""
    try:
        task_uuid = uuid.UUID(payload.task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task UUID format")

    # BOLA / IDOR isolation check
    task = await check_task_access(task_uuid, current_user, db)

    # Get rubric version
    rubric_version = payload.rubric_version
    if not rubric_version:
        rubric_result = await db.execute(
            select(TaskRubric).filter_by(task_id=task.id, is_active=True)
            .order_by(TaskRubric.created_at.desc()).limit(1)
        )
        rubric = rubric_result.scalar_one_or_none()
        if not rubric:
            raise HTTPException(status_code=400, detail="No active rubric found. Create one first.")
        rubric_version = rubric.version

    # Count parsed submissions
    sub_result = await db.execute(
        select(Submission).filter_by(task_id=task.id, status="PARSED")
    )
    submissions = sub_result.scalars().all()

    model = payload.model or settings.GEMINI_MODEL

    run = GradingRun(
        task_id=task.id,
        rubric_version=rubric_version,
        model=model,
        temperature=payload.temperature,
        description=payload.description,
        status="CREATED",
        total_submissions=len(submissions),
        graded_count=0,
        failed_count=0,
        created_by=current_user.id,
    )
    db.add(run)
    await db.flush()

    response = GradingRunResponse(
        id=str(run.id), task_id=str(run.task_id),
        rubric_version=run.rubric_version, model=run.model,
        temperature=run.temperature, description=run.description,
        status=run.status, total_submissions=run.total_submissions,
        graded_count=0, failed_count=0,
        created_at=run.created_at.isoformat(), completed_at=None,
    )
    return ApiResponse(data=response)


@router.post("/{run_id}/start")
async def start_run(run_id: str, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Start grading all parsed submissions for this run."""
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid grading run UUID format")

    # BOLA / IDOR isolation check
    run = await check_run_access(run_uuid, current_user, db)
    if run.status != "CREATED":
        raise HTTPException(status_code=400, detail=f"Run already in status: {run.status}")

    # Get all parsed submissions for this task
    sub_result = await db.execute(
        select(Submission).filter_by(task_id=run.task_id, status="PARSED")
    )
    submissions = sub_result.scalars().all()

    if not submissions:
        raise HTTPException(status_code=400, detail="No parsed submissions to grade")

    run.status = "RUNNING"
    run.total_submissions = len(submissions)
    await db.flush()

    # Enqueue grading tasks
    from app.tasks.grade_submission import grade, finalize_run
    from celery import chain, chord

    grade_tasks = [grade.s(str(sub.id), str(run.id)) for sub in submissions]
    callback = finalize_run.si(str(run.id))
    chord(grade_tasks)(callback)

    return ApiResponse(data={
        "run_id": str(run.id),
        "status": "RUNNING",
        "submissions_queued": len(submissions),
        "message": f"Grading started for {len(submissions)} submissions",
    })


@router.get("/{run_id}")
async def get_run(run_id: str, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Get grading run status and progress."""
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid grading run UUID format")

    # BOLA / IDOR isolation check
    run = await check_run_access(run_uuid, current_user, db)

    response = GradingRunResponse(
        id=str(run.id), task_id=str(run.task_id),
        rubric_version=run.rubric_version, model=run.model,
        temperature=run.temperature, description=run.description,
        status=run.status, total_submissions=run.total_submissions or 0,
        graded_count=run.graded_count or 0, failed_count=run.failed_count or 0,
        created_at=run.created_at.isoformat(),
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
    )
    return ApiResponse(data=response)


@router.get("/{run_id}/results")
async def get_run_results(run_id: str, page: int = 1, page_size: int = 20,
                          db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid grading run UUID format")

    # BOLA / IDOR isolation check
    await check_run_access(run_uuid, current_user, db)

    offset = (page - 1) * page_size
    result = await db.execute(
        select(GradeResult).filter_by(grading_run_id=run_uuid)
        .order_by(GradeResult.graded_at.desc())
        .offset(offset).limit(page_size)
    )
    grades = result.scalars().all()

    items = []
    for g in grades:
        step_grades = [StepGradeResult(**sg) for sg in (g.step_grades or [])]
        items.append(GradeResultResponse(
            id=str(g.id), submission_id=str(g.submission_id),
            grading_run_id=str(g.grading_run_id),
            grade=g.grade, max_grade=g.max_grade,
            grade_distribution=g.grade_distribution,
            confidence=g.confidence or 0.0, step_grades=step_grades,
            justification=g.justification, model_used=g.model_used,
            graded_at=g.graded_at.isoformat() if g.graded_at else "",
            latency_ms=g.latency_ms,
        ))
    return ApiResponse(data={"items": items, "page": page, "page_size": page_size})


@router.get("/{run_id}/statistics")
async def get_run_statistics(run_id: str, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid grading run UUID format")

    # BOLA / IDOR isolation check
    run = await check_run_access(run_uuid, current_user, db)

    grade_result = await db.execute(select(GradeResult).filter_by(grading_run_id=run_uuid))
    grades = grade_result.scalars().all()

    task_result = await db.execute(select(Task).filter_by(id=run.task_id))
    task = task_result.scalar_one_or_none()
    max_grade = task.max_marks if task else 20

    from app.services.drift_detector import compute_run_statistics
    grade_dicts = [
        {"grade_distribution": g.grade_distribution, "grade": g.grade,
         "confidence": g.confidence, "latency_ms": g.latency_ms,
         "step_grades": g.step_grades}
        for g in grades
    ]
    stats = compute_run_statistics(grade_dicts, max_grade)
    stats["run_id"] = str(run_id)

    return ApiResponse(data=RunStatistics(**stats))


@router.get("/{run_id}/drift")
async def get_run_drift(run_id: str, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid grading run UUID format")

    # BOLA / IDOR isolation check
    await check_run_access(run_uuid, current_user, db)

    result = await db.execute(
        select(DriftReport).filter_by(current_run_id=run_uuid)
        .order_by(DriftReport.created_at.desc()).limit(1)
    )
    report = result.scalar_one_or_none()

    if not report:
        # Try to compute drift on the fly
        run_result = await db.execute(select(GradingRun).filter_by(id=run_id))
        run = run_result.scalar_one_or_none()
        if not run:
            raise HTTPException(status_code=404, detail="Grading run not found")

        task_result = await db.execute(select(Task).filter_by(id=run.task_id))
        task = task_result.scalar_one_or_none()

        if not task or not task.baseline_run_id:
            raise HTTPException(status_code=404, detail="No baseline set for this task. Set one first.")

        raise HTTPException(status_code=404, detail="No drift report found. Run drift detection first.")

    response = DriftReportResponse(
        id=str(report.id), task_id=str(report.task_id),
        current_run_id=str(report.current_run_id),
        baseline_run_id=str(report.baseline_run_id),
        kl_divergence=report.kl_divergence, mean_shift=report.mean_shift,
        entropy_current=report.entropy_current, entropy_baseline=report.entropy_baseline,
        drift_detected=report.drift_detected, severity=report.severity or "LOW",
        details=report.details, created_at=report.created_at.isoformat(),
    )
    return ApiResponse(data=response)


import asyncio
import json
from fastapi.responses import StreamingResponse

@router.get("/{run_id}/events")
async def run_events_stream(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid grading run UUID format")

    # BOLA / IDOR isolation check
    await check_run_access(run_uuid, current_user, db)

    async def event_generator():
        while True:
            try:
                # Retrieve fresh status from DB
                result = await db.execute(select(GradingRun).filter_by(id=run_uuid))
                run = result.scalar_one_or_none()

                if not run:
                    yield f"data: {json.dumps({'error': 'Grading run not found'})}\n\n"
                    break

                total = run.total_submissions or 0
                graded = run.graded_count or 0
                failed = run.failed_count or 0
                processed = graded + failed

                progress = round((processed / total) * 100 if total > 0 else 0, 2)

                payload = {
                    "run_id": str(run.id),
                    "status": run.status,
                    "total_submissions": total,
                    "graded_count": graded,
                    "failed_count": failed,
                    "progress_percentage": progress,
                }

                yield f"data: {json.dumps(payload)}\n\n"

                if run.status in ["COMPLETED", "FAILED"] or (total > 0 and processed >= total):
                    break

                await asyncio.sleep(2)
                db.expire_all()
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


"""
Tasks API — CRUD for assessment tasks and rubrics.
"""
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import Task, TaskRubric, GradingAlert, User
from app.schemas.common import ApiResponse
from app.schemas.rubric import TaskCreate, TaskResponse, TaskRubricCreate, TaskRubricResponse
from app.schemas.observability import GradingAlertResponse, SetBaselineRequest
from app.services.auth_service import get_current_user

router = APIRouter(prefix="/tasks", tags=["Tasks"])


@router.post("")
async def create_task(payload: TaskCreate, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Create a new assessment task, optionally with an initial rubric."""
    task = Task(
        title=payload.title,
        subject=payload.subject,
        board=payload.board,
        grade_level=payload.grade_level,
        max_marks=payload.max_marks,
        description=payload.description,
    )
    db.add(task)
    await db.flush()

    # Create initial rubric if provided
    rubric_response = None
    if payload.rubric:
        rubric = TaskRubric(
            task_id=task.id,
            version=payload.rubric.version,
            rubric_json={
                "steps": [s.model_dump() for s in payload.rubric.steps],
            },
            grading_notes=payload.rubric.grading_notes,
            is_active=True,
        )
        db.add(rubric)
        await db.flush()
        rubric_response = TaskRubricResponse(
            id=str(rubric.id), task_id=str(task.id), version=rubric.version,
            steps=payload.rubric.steps, grading_notes=rubric.grading_notes,
            is_active=True, created_at=rubric.created_at.isoformat(),
        )

    response = TaskResponse(
        id=str(task.id), title=task.title, subject=task.subject,
        board=task.board, grade_level=task.grade_level,
        max_marks=task.max_marks, description=task.description,
        baseline_run_id=None,
        created_at=task.created_at.isoformat(), updated_at=task.updated_at.isoformat(),
        current_rubric=rubric_response,
    )
    return ApiResponse(data=response)


@router.get("/{task_id}")
async def get_task(task_id: str, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Get a task with its current active rubric."""
    result = await db.execute(select(Task).filter_by(id=task_id))
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Get current active rubric
    rubric_result = await db.execute(
        select(TaskRubric).filter_by(task_id=task.id, is_active=True)
        .order_by(TaskRubric.created_at.desc()).limit(1)
    )
    rubric = rubric_result.scalar_one_or_none()

    rubric_response = None
    if rubric:
        steps_data = rubric.rubric_json.get("steps", [])
        rubric_response = TaskRubricResponse(
            id=str(rubric.id), task_id=str(task.id), version=rubric.version,
            steps=steps_data, grading_notes=rubric.grading_notes,
            is_active=rubric.is_active, created_at=rubric.created_at.isoformat(),
        )

    response = TaskResponse(
        id=str(task.id), title=task.title, subject=task.subject,
        board=task.board, grade_level=task.grade_level,
        max_marks=task.max_marks, description=task.description,
        baseline_run_id=str(task.baseline_run_id) if task.baseline_run_id else None,
        created_at=task.created_at.isoformat(), updated_at=task.updated_at.isoformat(),
        current_rubric=rubric_response,
    )
    return ApiResponse(data=response)


@router.get("")
async def list_tasks(db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    """List all tasks."""
    result = await db.execute(select(Task).order_by(Task.created_at.desc()))
    tasks = result.scalars().all()

    items = [
        {"id": str(t.id), "title": t.title, "subject": t.subject,
         "board": t.board, "max_marks": t.max_marks, "created_at": t.created_at.isoformat()}
        for t in tasks
    ]
    return ApiResponse(data=items)


@router.post("/{task_id}/rubrics")
async def create_rubric(task_id: str, payload: TaskRubricCreate, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Upload a new rubric version for a task."""
    result = await db.execute(select(Task).filter_by(id=task_id))
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Deactivate previous rubrics
    prev_result = await db.execute(select(TaskRubric).filter_by(task_id=task_id, is_active=True))
    for prev in prev_result.scalars().all():
        prev.is_active = False

    rubric = TaskRubric(
        task_id=task.id,
        version=payload.version,
        rubric_json={"steps": [s.model_dump() for s in payload.steps]},
        grading_notes=payload.grading_notes,
        is_active=True,
    )
    db.add(rubric)
    await db.flush()

    response = TaskRubricResponse(
        id=str(rubric.id), task_id=str(task.id), version=rubric.version,
        steps=payload.steps, grading_notes=rubric.grading_notes,
        is_active=True, created_at=rubric.created_at.isoformat(),
    )
    return ApiResponse(data=response)


@router.post("/{task_id}/baseline")
async def set_baseline(task_id: str, payload: SetBaselineRequest, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Set a grading run as the drift baseline for a task."""
    result = await db.execute(select(Task).filter_by(id=task_id))
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    task.baseline_run_id = uuid.UUID(payload.run_id)
    return ApiResponse(data={"task_id": str(task.id), "baseline_run_id": payload.run_id})


@router.get("/{task_id}/alerts")
async def get_task_alerts(task_id: str, resolved: bool = False, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Get active alerts for a task."""
    query = select(GradingAlert).filter_by(task_id=task_id, resolved=resolved).order_by(GradingAlert.created_at.desc())
    result = await db.execute(query)
    alerts = result.scalars().all()

    items = [
        GradingAlertResponse(
            id=str(a.id), run_id=str(a.run_id) if a.run_id else None,
            task_id=str(a.task_id) if a.task_id else None,
            alert_type=a.alert_type, severity=a.severity, message=a.message,
            metadata_json=a.metadata_json, resolved=a.resolved,
            created_at=a.created_at.isoformat(),
        )
        for a in alerts
    ]
    return ApiResponse(data=items)

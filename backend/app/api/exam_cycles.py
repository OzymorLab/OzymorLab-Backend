"""
Exam Cycles API — Create, list, update, and view exam cycles.

An ExamCycle groups multiple subject papers (Tasks) into a single exam event.
Access is always scoped by the current user's school_id for tenant isolation.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_

from app.db.session import get_db
from app.db.models import ExamCycle, Task, User
from app.schemas.common import ApiResponse
from app.schemas.operations import (
    ExamCycleCreate,
    ExamCycleUpdate,
    ExamCycleResponse,
    ExamCycleDetailResponse,
)
from app.services.auth_service import get_current_user, require_role

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/exam-cycles",
    tags=["Exam Cycles"],
)


@router.post("", dependencies=[Depends(require_role(["teacher", "admin", "hod", "principal", "student"]))])
async def create_exam_cycle(
    payload: ExamCycleCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new exam cycle for the current user (associated with school if assigned)."""
    # Validate date ordering
    if payload.start_date and payload.end_date and payload.end_date <= payload.start_date:
        raise HTTPException(status_code=400, detail="end_date must be after start_date.")

    cycle = ExamCycle(
        school_id=current_user.school_id,
        name=payload.name,
        start_date=payload.start_date,
        end_date=payload.end_date,
        status="ACTIVE",
        created_by=current_user.id,
    )
    db.add(cycle)
    await db.flush()

    logger.info(f"ExamCycle created: {cycle.id} by user {current_user.id}")

    return ApiResponse(data=ExamCycleResponse(
        id=str(cycle.id),
        school_id=str(cycle.school_id) if cycle.school_id else None,
        name=cycle.name,
        start_date=cycle.start_date.isoformat() if cycle.start_date else None,
        end_date=cycle.end_date.isoformat() if cycle.end_date else None,
        status=cycle.status,
        created_by=str(cycle.created_by),
        created_at=cycle.created_at.isoformat(),
        task_count=0,
    ))


@router.get("", dependencies=[Depends(require_role(["teacher", "admin", "hod", "principal", "student"]))])
async def list_exam_cycles(
    status: str | None = None,
    limit: int = Query(50, ge=1, le=100),
    skip: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List exam cycles scoped to the current user's school or created independently."""
    if current_user.school_id:
        query = (
            select(ExamCycle)
            .filter(or_(ExamCycle.school_id == current_user.school_id, ExamCycle.created_by == current_user.id))
            .order_by(ExamCycle.created_at.desc())
        )
    else:
        query = (
            select(ExamCycle)
            .filter(ExamCycle.created_by == current_user.id)
            .order_by(ExamCycle.created_at.desc())
        )
        
    if status:
        query = query.filter(ExamCycle.status == status)
    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    cycles = result.scalars().all()

    items = []
    for c in cycles:
        # Count linked tasks
        task_count_result = await db.execute(
            select(func.count(Task.id)).filter_by(exam_cycle_id=c.id)
        )
        task_count = task_count_result.scalar() or 0

        items.append(ExamCycleResponse(
            id=str(c.id),
            school_id=str(c.school_id) if c.school_id else None,
            name=c.name,
            start_date=c.start_date.isoformat() if c.start_date else None,
            end_date=c.end_date.isoformat() if c.end_date else None,
            status=c.status,
            created_by=str(c.created_by),
            created_at=c.created_at.isoformat(),
            task_count=task_count,
        ))

    return ApiResponse(data=items)


@router.get("/{cycle_id}", dependencies=[Depends(require_role(["teacher", "admin", "hod", "principal", "student"]))])
async def get_exam_cycle(
    cycle_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get exam cycle details with linked tasks."""
    if current_user.school_id:
        result = await db.execute(
            select(ExamCycle).filter(
                ExamCycle.id == cycle_id,
                or_(ExamCycle.school_id == current_user.school_id, ExamCycle.created_by == current_user.id)
            )
        )
    else:
        result = await db.execute(
            select(ExamCycle).filter(
                ExamCycle.id == cycle_id,
                ExamCycle.created_by == current_user.id
            )
        )
    cycle = result.scalar_one_or_none()
    if not cycle:
        raise HTTPException(status_code=404, detail="Exam cycle not found.")

    # Fetch linked tasks
    tasks_result = await db.execute(
        select(Task).filter_by(exam_cycle_id=cycle.id).order_by(Task.subject)
    )
    tasks = tasks_result.scalars().all()

    task_items = [
        {
            "id": str(t.id),
            "title": t.title,
            "subject": t.subject,
            "board": t.board,
            "grade_level": t.grade_level,
            "max_marks": t.max_marks,
            "paper_set": t.paper_set,
            "created_at": t.created_at.isoformat(),
        }
        for t in tasks
    ]

    return ApiResponse(data=ExamCycleDetailResponse(
        id=str(cycle.id),
        school_id=str(cycle.school_id) if cycle.school_id else None,
        name=cycle.name,
        start_date=cycle.start_date.isoformat() if cycle.start_date else None,
        end_date=cycle.end_date.isoformat() if cycle.end_date else None,
        status=cycle.status,
        created_by=str(cycle.created_by),
        created_at=cycle.created_at.isoformat(),
        task_count=len(task_items),
        tasks=task_items,
    ))


@router.patch("/{cycle_id}", dependencies=[Depends(require_role(["teacher", "admin", "hod", "principal", "student"]))])
async def update_exam_cycle(
    cycle_id: str,
    payload: ExamCycleUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update exam cycle metadata."""
    if current_user.school_id:
        result = await db.execute(
            select(ExamCycle).filter(
                ExamCycle.id == cycle_id,
                or_(ExamCycle.school_id == current_user.school_id, ExamCycle.created_by == current_user.id)
            )
        )
    else:
        result = await db.execute(
            select(ExamCycle).filter(
                ExamCycle.id == cycle_id,
                ExamCycle.created_by == current_user.id
            )
        )
    cycle = result.scalar_one_or_none()
    if not cycle:
        raise HTTPException(status_code=404, detail="Exam cycle not found.")

    if payload.name is not None:
        cycle.name = payload.name
    if payload.start_date is not None:
        cycle.start_date = payload.start_date
    if payload.end_date is not None:
        cycle.end_date = payload.end_date
    if payload.status is not None:
        cycle.status = payload.status

    # Re-validate date ordering after update
    if cycle.start_date and cycle.end_date and cycle.end_date <= cycle.start_date:
        raise HTTPException(status_code=400, detail="end_date must be after start_date.")

    await db.flush()
    logger.info(f"ExamCycle updated: {cycle.id} by user {current_user.id}")

    return ApiResponse(data=ExamCycleResponse(
        id=str(cycle.id),
        school_id=str(cycle.school_id) if cycle.school_id else None,
        name=cycle.name,
        start_date=cycle.start_date.isoformat() if cycle.start_date else None,
        end_date=cycle.end_date.isoformat() if cycle.end_date else None,
        status=cycle.status,
        created_by=str(cycle.created_by),
        created_at=cycle.created_at.isoformat(),
    ))

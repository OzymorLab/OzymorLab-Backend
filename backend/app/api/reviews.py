"""
Reviews API — Human moderation endpoints for the multimodal evaluation system.

Provides endpoints for:
  - Listing submissions flagged for review (NEEDS_REVIEW).
  - Viewing detailed component breakdowns for a submission.
  - Approving an AI-generated grade.
  - Overriding a grade with a teacher's assessment + moderation notes.
"""
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import GradeResult, Submission, Task, User, GradingRun, ExamCycle
from app.schemas.common import ApiResponse
from app.services.auth_service import get_current_user, require_role

router = APIRouter(
    prefix="/reviews", 
    tags=["Reviews"],
    dependencies=[Depends(require_role(["teacher", "admin", "hod", "principal"]))]
)


def _scope_review_query_by_role(query, user: User):
    """
    Apply role-based filtering to review queries for tenant isolation.

    - teacher: Only sees reviews for tasks whose grading runs they created.
    - hod: Sees reviews for all tasks in their school.
    - principal/admin: Sees all reviews in their school.
    - No school_id: Sees all reviews (legacy/standalone mode).
    """
    if not user.school_id:
        # Legacy standalone teacher — no school scoping
        return query

    if user.role == "teacher":
        # Teacher only sees reviews for tasks they triggered grading on
        query = (
            query
            .join(GradingRun, GradeResult.grading_run_id == GradingRun.id)
            .filter(GradingRun.created_by == user.id)
        )
    elif user.role in ("hod", "principal", "admin"):
        # HOD/Principal/Admin sees reviews for tasks in their school
        # Scope via exam_cycle → school, or via the grading run creator's school
        query = (
            query
            .join(GradingRun, GradeResult.grading_run_id == GradingRun.id)
            .outerjoin(User, GradingRun.created_by == User.id)
            .filter(User.school_id == user.school_id)
        )

    return query


# ── Request/Response Schemas ──

class ReviewOverrideRequest(BaseModel):
    """Schema for overriding a grade during human review."""
    grade: int = Field(ge=0, description="Teacher's corrected grade")
    notes: str = Field(min_length=1, description="Moderation notes explaining the override")
    reviewer_id: str = Field(min_length=1, description="Teacher/evaluator identifier")


class ReviewApproveRequest(BaseModel):
    """Schema for approving an AI-generated grade."""
    reviewer_id: str = Field(min_length=1, description="Teacher/evaluator identifier")
    notes: str = ""


class ReviewItemResponse(BaseModel):
    """Summary of a submission needing review."""
    submission_id: str
    student_id: str
    task_title: str
    grade: int
    max_grade: int
    confidence: float
    review_status: str
    review_reasons: list[str] | None
    flagged_components: list[str] | None
    graded_at: str | None

    model_config = {"from_attributes": True}


class ReviewDetailResponse(BaseModel):
    """Detailed component breakdown for review."""
    submission_id: str
    student_id: str
    task_title: str
    grade: int
    max_grade: int
    confidence: float
    review_status: str
    review_reasons: list[str] | None
    flagged_components: list[str] | None
    component_grades: list[dict] | None
    step_grades: list[dict] | None
    justification: str | None
    review_notes: str | None
    reviewed_by: str | None

    model_config = {"from_attributes": True}


# ── Endpoints ──

@router.get("/pending")
async def list_pending_reviews(
    task_id: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List all submissions flagged for human review.
    Results are scoped by the current user's role:
    - Teacher: only their own tasks' reviews.
    - HOD/Principal/Admin: all reviews in their school.
    Optionally filter by task_id.
    """
    query = (
        select(GradeResult, Submission, Task)
        .join(Submission, GradeResult.submission_id == Submission.id)
        .join(Task, Submission.task_id == Task.id)
        .filter(GradeResult.review_status == "NEEDS_REVIEW")
        .order_by(GradeResult.graded_at.desc())
    )

    if task_id:
        query = query.filter(Submission.task_id == task_id)

    # Apply role-based scoping
    query = _scope_review_query_by_role(query, current_user)

    result = await db.execute(query)
    rows = result.all()

    items = []
    for grade_result, submission, task in rows:
        items.append(ReviewItemResponse(
            submission_id=str(submission.id),
            student_id=str(submission.student_id) if submission.student_id else "N/A",
            task_title=task.title,
            grade=grade_result.grade,
            max_grade=grade_result.max_grade,
            confidence=grade_result.confidence or 0.0,
            review_status=grade_result.review_status,
            review_reasons=grade_result.review_reasons,
            flagged_components=grade_result.flagged_components,
            graded_at=grade_result.graded_at.isoformat() if grade_result.graded_at else None,
        ))

    return {"data": items, "total": len(items)}


@router.get("/{submission_id}")
async def get_review_detail(submission_id: str, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Get detailed component breakdown for a specific submission.
    Used by the teacher during moderation.
    """
    result = await db.execute(
        select(GradeResult, Submission, Task)
        .join(Submission, GradeResult.submission_id == Submission.id)
        .join(Task, Submission.task_id == Task.id)
        .filter(Submission.id == submission_id)
        .order_by(GradeResult.graded_at.desc())
        .limit(1)
    )
    row = result.first()

    if not row:
        raise HTTPException(status_code=404, detail="Submission or grade result not found")

    grade_result, submission, task = row

    return {
        "data": ReviewDetailResponse(
            submission_id=str(submission.id),
            student_id=submission.student_id,
            task_title=task.title,
            grade=grade_result.grade,
            max_grade=grade_result.max_grade,
            confidence=grade_result.confidence or 0.0,
            review_status=grade_result.review_status,
            review_reasons=grade_result.review_reasons,
            flagged_components=grade_result.flagged_components,
            component_grades=grade_result.component_grades,
            step_grades=grade_result.step_grades,
            justification=grade_result.justification,
            review_notes=grade_result.review_notes,
            reviewed_by=grade_result.reviewed_by,
        )
    }


@router.post("/{submission_id}/approve")
async def approve_grade(
    submission_id: str,
    payload: ReviewApproveRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Approve the AI-generated grade for a submission.
    Moves the review_status from NEEDS_REVIEW → REVIEWED.
    """
    result = await db.execute(
        select(GradeResult)
        .join(Submission, GradeResult.submission_id == Submission.id)
        .filter(Submission.id == submission_id)
        .order_by(GradeResult.graded_at.desc())
        .limit(1)
    )
    grade_result = result.scalar_one_or_none()

    if not grade_result:
        raise HTTPException(status_code=404, detail="Grade result not found")

    grade_result.review_status = "REVIEWED"
    grade_result.reviewed_by = payload.reviewer_id
    grade_result.review_notes = payload.notes or "Approved by human reviewer"
    grade_result.reviewed_at = datetime.now(timezone.utc)

    await db.commit()

    return {
        "data": {
            "submission_id": submission_id,
            "review_status": "REVIEWED",
            "reviewed_by": payload.reviewer_id,
            "message": "Grade approved successfully",
        }
    }


@router.post("/{submission_id}/override")
async def override_grade(
    submission_id: str,
    payload: ReviewOverrideRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Override the AI-generated grade with a teacher's corrected grade.
    Moves the review_status to OVERRIDDEN.
    """
    result = await db.execute(
        select(GradeResult)
        .join(Submission, GradeResult.submission_id == Submission.id)
        .filter(Submission.id == submission_id)
        .order_by(GradeResult.graded_at.desc())
        .limit(1)
    )
    grade_result = result.scalar_one_or_none()

    if not grade_result:
        raise HTTPException(status_code=404, detail="Grade result not found")

    if payload.grade > grade_result.max_grade:
        raise HTTPException(
            status_code=400,
            detail=f"Override grade ({payload.grade}) cannot exceed max grade ({grade_result.max_grade})"
        )

    # Store original grade before override
    original_grade = grade_result.grade

    grade_result.grade = payload.grade
    grade_result.review_status = "OVERRIDDEN"
    grade_result.reviewed_by = payload.reviewer_id
    grade_result.review_notes = payload.notes
    grade_result.reviewed_at = datetime.now(timezone.utc)

    await db.commit()

    return {
        "data": {
            "submission_id": submission_id,
            "original_grade": original_grade,
            "overridden_grade": payload.grade,
            "review_status": "OVERRIDDEN",
            "reviewed_by": payload.reviewer_id,
            "message": "Grade overridden successfully",
        }
    }

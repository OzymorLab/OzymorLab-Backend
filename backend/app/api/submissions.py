"""
Submissions API — file upload, status tracking, grade retrieval.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import Submission, GradeResult, Task
from app.schemas.common import ApiResponse
from app.schemas.submission import SubmissionResponse, SubmissionUploadResponse, ParsedContent
from app.schemas.grade import StepGradeResult, GradeResultResponse
from app.services.ingestion import validate_file, upload_file

router = APIRouter(prefix="/submissions", tags=["Submissions"])


@router.post("")
async def create_submission(
    task_id: str = Form(...),
    student_id: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a student submission file → store in MinIO → queue for parsing."""
    # Validate task exists
    result = await db.execute(select(Task).filter_by(id=task_id))
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Read and validate file
    file_data = await file.read()
    filename = file.filename or "upload.pdf"

    is_valid, error_msg = validate_file(filename, len(file_data))
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    # Upload to MinIO
    content_type = file.content_type or "application/octet-stream"
    file_key = upload_file(file_data, filename, content_type)

    # Determine file type
    file_type = filename.rsplit(".", 1)[-1].lower() if "." in filename else "pdf"

    # Create submission record
    submission = Submission(
        task_id=task.id,
        student_id=student_id,
        file_key=file_key,
        file_name=filename,
        file_type=file_type,
        status="PENDING",
    )
    db.add(submission)
    await db.flush()

    # Enqueue Celery parse task
    from app.tasks.parse_submission import parse
    parse.delay(str(submission.id))

    return ApiResponse(data=SubmissionUploadResponse(
        submission_id=str(submission.id),
        status="PENDING",
        message="Submission queued for parsing",
    ))


@router.get("/{submission_id}")
async def get_submission(submission_id: str, db: AsyncSession = Depends(get_db)):
    """Get submission status and parsed content."""
    result = await db.execute(select(Submission).filter_by(id=submission_id))
    submission = result.scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    parsed = None
    if submission.parsed_content:
        parsed = ParsedContent(**submission.parsed_content)

    response = SubmissionResponse(
        id=str(submission.id),
        task_id=str(submission.task_id),
        student_id=submission.student_id,
        file_name=submission.file_name,
        file_type=submission.file_type,
        status=submission.status,
        raw_text=submission.raw_text,
        parsed_content=parsed,
        error_message=submission.error_message,
        created_at=submission.created_at.isoformat(),
        updated_at=submission.updated_at.isoformat(),
    )
    return ApiResponse(data=response)


@router.get("")
async def list_submissions(
    task_id: str | None = None,
    status: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List submissions, optionally filtered by task_id and/or status."""
    query = select(Submission).order_by(Submission.created_at.desc())
    if task_id:
        query = query.filter_by(task_id=task_id)
    if status:
        query = query.filter_by(status=status)

    result = await db.execute(query)
    submissions = result.scalars().all()

    items = [
        {"id": str(s.id), "task_id": str(s.task_id), "student_id": s.student_id,
         "file_name": s.file_name, "status": s.status, "created_at": s.created_at.isoformat()}
        for s in submissions
    ]
    return ApiResponse(data=items)


@router.get("/{submission_id}/grade")
async def get_submission_grade(submission_id: str, db: AsyncSession = Depends(get_db)):
    """Get the grade result for a submission with full step trace."""
    result = await db.execute(
        select(GradeResult).filter_by(submission_id=submission_id)
        .order_by(GradeResult.graded_at.desc()).limit(1)
    )
    grade = result.scalar_one_or_none()
    if not grade:
        raise HTTPException(status_code=404, detail="No grade result found for this submission")

    step_grades = [StepGradeResult(**sg) for sg in (grade.step_grades or [])]

    response = GradeResultResponse(
        id=str(grade.id),
        submission_id=str(grade.submission_id),
        grading_run_id=str(grade.grading_run_id),
        grade=grade.grade,
        max_grade=grade.max_grade,
        grade_distribution=grade.grade_distribution,
        confidence=grade.confidence or 0.0,
        step_grades=step_grades,
        justification=grade.justification,
        model_used=grade.model_used,
        graded_at=grade.graded_at.isoformat() if grade.graded_at else "",
        latency_ms=grade.latency_ms,
    )
    return ApiResponse(data=response)

"""
Submissions API — file upload, bulk upload, status tracking, grade retrieval.
"""
import json
import re
from typing import List, Optional

import csv
import io
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

import uuid
from app.db.session import get_db
from app.db.models import Submission, GradeResult, GradingRun, Task, TaskRubric, User, Student
from app.schemas.common import ApiResponse
from app.schemas.submission import SubmissionResponse, SubmissionUploadResponse, ParsedContent
from app.schemas.grade import StepGradeResult, GradeResultResponse
from app.services.ingestion import validate_file, upload_file
from app.services.auth_service import (
    get_current_user,
    require_role,
    check_task_access,
    check_submission_access
)
from app.config import settings

router = APIRouter(
    prefix="/submissions", 
    tags=["Submissions"],
    dependencies=[Depends(require_role(["teacher", "admin", "hod", "principal", "student"]))]
)


def is_valid_uuid(val: str) -> bool:
    try:
        import uuid
        uuid.UUID(str(val))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


@router.post("")
async def create_submission(
    request: Request,
    task_id: str = Form(...),
    student_id: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload a student submission file → store in S3 → queue for parsing."""
    # BOLA / IDOR isolation check
    try:
        task_uuid = uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task UUID format")
    task = await check_task_access(task_uuid, current_user, db)

    # Read and validate file
    file_data = await file.read()
    filename = file.filename or "upload.pdf"

    is_valid, error_msg = validate_file(filename, len(file_data))
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    # Upload to S3
    content_type = file.content_type or "application/octet-stream"
    user_token = request.headers.get("Authorization")
    file_key = upload_file(file_data, filename, content_type, user_token=user_token)

    # Determine file type
    file_type = filename.rsplit(".", 1)[-1].lower() if "." in filename else "pdf"

    # Clean up student_id if it's the legacy dummy string
    if student_id and not student_id.replace('-', '').isalnum() and len(student_id) < 30:
        student_id = None # Ignore dummy strings, let OCR handle it

    # Check if student_id is a valid UUID
    db_student_id = None
    if student_id:
        if is_valid_uuid(student_id):
            db_student_id = student_id
        else:
            # Try to lookup student
            clean_term = student_id
            if clean_term.upper().startswith("STUDENT-"):
                clean_term = clean_term[8:]
            student_result = await db.execute(
                select(Student).filter(
                    (Student.roll_number.ilike(clean_term)) |
                    (Student.name.ilike(clean_term)) |
                    (Student.roll_number.ilike(student_id)) |
                    (Student.name.ilike(student_id))
                )
            )
            student_obj = student_result.scalar_one_or_none()
            if student_obj:
                db_student_id = str(student_obj.id)

    # Create submission record
    submission = Submission(
        task_id=task.id,
        student_id=db_student_id,
        file_key=file_key,
        file_name=filename,
        file_type=file_type,
        status="PENDING",
    )
    db.add(submission)
    await db.flush()

    # Enqueue Celery Orchestrator task (auto-grading pipeline)
    from app.tasks.orchestrator import process_submission
    process_submission.delay(str(submission.id))

    return ApiResponse(data=SubmissionUploadResponse(
        submission_id=str(submission.id),
        status="PENDING",
        message="Submission queued for auto-grading pipeline",
    ))


@router.get("/{submission_id}")
async def get_submission(submission_id: str, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Get submission status and parsed content."""
    try:
        sub_uuid = uuid.UUID(submission_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid submission UUID format")

    # BOLA / IDOR isolation check
    await check_submission_access(sub_uuid, current_user, db)

    result = await db.execute(select(Submission).filter_by(id=sub_uuid))
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


@router.get("/export")
async def export_submissions_csv(
    task_id: str = Query(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Export graded submissions as a CSV file."""
    try:
        task_uuid = uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task UUID format")

    # BOLA / IDOR isolation check
    await check_task_access(task_uuid, current_user, db)

    # Query graded submissions for a specific task and eager load grade_results
    from sqlalchemy.orm import selectinload
    query = (
        select(Submission)
        .options(selectinload(Submission.grade_results))
        .filter_by(task_id=task_uuid, status="GRADED")
        .order_by(Submission.created_at.desc())
    )
    result = await db.execute(query)
    submissions = result.scalars().all()

    # Generate CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Submission ID", "Student ID", "File Name", "Status", "Total Marks", "Created At"])

    for s in submissions:
        total_marks = sum(r.marks_awarded for r in s.grade_results) if s.grade_results else 0.0
        writer.writerow([str(s.id), str(s.student_id), s.file_name, s.status, total_marks, s.created_at.isoformat()])

    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=export_task_{task_id}.csv"}
    )


@router.get("")
async def list_submissions(
    task_id: str | None = None,
    status: str | None = None,
    limit: int = Query(50, ge=1, le=100),
    skip: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List submissions, optionally filtered by task_id and/or status."""
    query = select(Submission).order_by(Submission.created_at.desc())
    if task_id:
        try:
            task_uuid = uuid.UUID(task_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid task UUID format")
        
        # BOLA / IDOR isolation check
        await check_task_access(task_uuid, current_user, db)
        query = query.filter_by(task_id=task_uuid)
    
    # Enforce student-level isolation for the list endpoint as well
    if current_user.role == "student":
        from sqlalchemy import func
        email_name = current_user.email.split("@")[0].replace(".", " ").title()
        student_stmt = select(Student).filter(
            (Student.id == current_user.id) |
            (func.lower(Student.name) == func.lower(current_user.full_name)) |
            (func.lower(Student.name) == func.lower(email_name))
        )
        student_res = await db.execute(student_stmt)
        students = student_res.scalars().all()
        student_ids = [s.id for s in students]
        student_ids.append(current_user.id)
        query = query.filter(Submission.student_id.in_(student_ids))

    if status:
        query = query.filter_by(status=status)
        
    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    submissions = result.scalars().all()

    items = [
        {"id": str(s.id), "task_id": str(s.task_id), "student_id": str(s.student_id) if s.student_id else None,
         "file_name": s.file_name, "status": s.status, "created_at": s.created_at.isoformat()}
        for s in submissions
    ]
    return ApiResponse(data=items)


@router.get("/{submission_id}/grade")
async def get_submission_grade(submission_id: str, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Get the grade result for a submission with full step trace."""
    try:
        sub_uuid = uuid.UUID(submission_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid submission UUID format")

    # BOLA / IDOR isolation check
    await check_submission_access(sub_uuid, current_user, db)

    result = await db.execute(
        select(GradeResult).filter_by(submission_id=sub_uuid)
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


# ── Bulk Upload ──

def _generate_student_id_from_filename(filename: str, index: int) -> str:
    """
    Generate a student ID from the filename.
    Examples:
        'physics_answer_rahul.pdf' → 'STUDENT-RAHUL'
        'answer_sheet_42.jpg' → 'STUDENT-42'
        'scan001.pdf' → 'STUDENT-001'
    Falls back to index-based ID if no useful text found.
    """
    # Remove extension
    name = filename.rsplit(".", 1)[0] if "." in filename else filename

    # Remove common prefixes
    for prefix in ["answer_sheet_", "answer_", "submission_", "scan", "sheet_"]:
        if name.lower().startswith(prefix):
            name = name[len(prefix):]

    # Clean up: remove underscores/hyphens, take the meaningful part
    name = name.strip("_- ")

    if name and len(name) <= 50:
        # Sanitize: keep only alphanumeric and hyphens
        clean = re.sub(r"[^a-zA-Z0-9\-]", "-", name).strip("-").upper()
        if clean:
            return f"STUDENT-{clean}"

    return f"STUDENT-{index + 1:03d}"


@router.post("/bulk")
async def bulk_upload_submissions(
    request: Request,
    task_id: str = Form(...),
    files: List[UploadFile] = File(...),
    student_ids: str = Form(""),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload multiple student answer sheets at once → store in S3 → queue all for parsing.

    Accepts up to 100 files per batch. If student_ids is provided (JSON array or
    comma-separated), maps them 1:1 to files. Otherwise, auto-generates IDs from filenames.
    """
    # Validate batch size
    if len(files) > 100:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files ({len(files)}). Maximum is 100 per batch."
        )

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    # BOLA / IDOR isolation check
    try:
        task_uuid = uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task UUID format")
    task = await check_task_access(task_uuid, current_user, db)

    # Parse student IDs
    sid_list: list[str] = []
    if student_ids and student_ids.strip():
        raw = student_ids.strip()
        # Try JSON array first
        if raw.startswith("["):
            try:
                sid_list = json.loads(raw)
            except json.JSONDecodeError:
                sid_list = [s.strip() for s in raw.split(",") if s.strip()]
        else:
            sid_list = [s.strip() for s in raw.split(",") if s.strip()]

    submitted = []
    failed = []

    user_token = request.headers.get("Authorization")

    for idx, upload_file_obj in enumerate(files):
        file_data = await upload_file_obj.read()
        filename = upload_file_obj.filename or f"upload_{idx}.pdf"

        # Validate file
        is_valid, error_msg = validate_file(filename, len(file_data))
        if not is_valid:
            failed.append({"file": filename, "error": error_msg})
            continue

        # Determine student ID
        if idx < len(sid_list) and sid_list[idx]:
            student_id = sid_list[idx]
        else:
            student_id = _generate_student_id_from_filename(filename, idx)

        # Upload to S3
        content_type = upload_file_obj.content_type or "application/octet-stream"
        try:
            file_key = upload_file(file_data, filename, content_type, user_token=user_token)
        except Exception as e:
            failed.append({"file": filename, "error": f"S3 upload failed: {str(e)}"})
            continue

        # Determine file type
        file_type = filename.rsplit(".", 1)[-1].lower() if "." in filename else "pdf"

        # Check if student_id is a valid UUID
        db_student_id = None
        if student_id:
            if is_valid_uuid(student_id):
                db_student_id = student_id
            else:
                # Try to lookup student
                clean_term = student_id
                if clean_term.upper().startswith("STUDENT-"):
                    clean_term = clean_term[8:]
                student_result = await db.execute(
                    select(Student).filter(
                        (Student.roll_number.ilike(clean_term)) |
                        (Student.name.ilike(clean_term)) |
                        (Student.roll_number.ilike(student_id)) |
                        (Student.name.ilike(student_id))
                    )
                )
                student_obj = student_result.scalar_one_or_none()
                if student_obj:
                    db_student_id = str(student_obj.id)

        # Create submission record
        submission = Submission(
            task_id=task.id,
            student_id=db_student_id,
            file_key=file_key,
            file_name=filename,
            file_type=file_type,
            status="PENDING",
        )
        db.add(submission)
        await db.flush()

        # Queue Celery parse task
        from app.tasks.parse_submission import parse
        parse.delay(str(submission.id))

        submitted.append({
            "submission_id": str(submission.id),
            "student_id": student_id,
            "file_name": filename,
        })

    return ApiResponse(data={
        "submitted": len(submitted),
        "failed": len(failed),
        "total_files": len(files),
        "submissions": submitted,
        "errors": failed,
        "message": f"{len(submitted)} answer sheets queued for parsing"
            + (f", {len(failed)} failed" if failed else ""),
    })


class BulkGradeRequest(BaseModel):
    """Request to start grading all parsed submissions for a task."""
    task_id: str
    description: str = ""
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


from fastapi import Request
from app.utils.idempotency import idempotent

@router.post("/bulk-grade")
@idempotent()
async def bulk_grade_submissions(
    payload: BulkGradeRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Convenience endpoint: create a grading run and immediately start grading
    all PARSED submissions for the given task. Combines POST /runs + POST /runs/{id}/start.
    """
    # BOLA / IDOR isolation check
    try:
        task_uuid = uuid.UUID(payload.task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task UUID format")
    task = await check_task_access(task_uuid, current_user, db)

    # Get active rubric
    rubric_result = await db.execute(
        select(TaskRubric).filter_by(task_id=task.id, is_active=True)
        .order_by(TaskRubric.created_at.desc()).limit(1)
    )
    rubric = rubric_result.scalar_one_or_none()
    if not rubric:
        raise HTTPException(status_code=400, detail="No active rubric found for this task.")

    # Phase 4: Rubric approval gate — only APPROVED rubrics can be used for grading
    if rubric.approval_status != "APPROVED":
        raise HTTPException(
            status_code=400,
            detail=f"Rubric is '{rubric.approval_status}'. Only APPROVED rubrics can be used for grading. "
                   f"Submit the rubric for HOD approval first.",
        )

    # Count parsed submissions
    sub_result = await db.execute(
        select(Submission).filter_by(task_id=task.id, status="PARSED")
    )
    submissions = sub_result.scalars().all()

    if not submissions:
        # Check if there are any pending/parsing submissions
        pending_result = await db.execute(
            select(Submission).filter_by(task_id=task.id)
            .filter(Submission.status.in_(["PENDING", "PARSING"]))
        )
        pending = pending_result.scalars().all()
        if pending:
            raise HTTPException(
                status_code=400,
                detail=f"{len(pending)} submissions are still being parsed. Wait for parsing to complete."
            )
        raise HTTPException(status_code=400, detail="No parsed submissions to grade.")

    # Create grading run
    model = settings.GEMINI_MODEL
    run = GradingRun(
        task_id=task.id,
        rubric_version=rubric.version,
        model=model,
        temperature=payload.temperature,
        description=payload.description or f"Bulk grading - {len(submissions)} submissions",
        status="RUNNING",
        total_submissions=len(submissions),
        graded_count=0,
        failed_count=0,
        created_by=current_user.id,
    )
    db.add(run)
    await db.flush()

    # Enqueue grading tasks
    from app.tasks.grade_submission import grade, finalize_run
    from celery import chord

    grade_tasks = [grade.s(str(sub.id), str(run.id)) for sub in submissions]
    callback = finalize_run.si(str(run.id))
    chord(grade_tasks)(callback)

    return ApiResponse(data={
        "run_id": str(run.id),
        "task_id": str(task.id),
        "status": "RUNNING",
        "submissions_queued": len(submissions),
        "rubric_version": rubric.version,
        "message": f"Grading started for {len(submissions)} submissions",
    })


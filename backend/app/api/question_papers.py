"""
Question Papers API — Upload question paper → AI rubric generation → Task creation.

Provides endpoints for:
  - Uploading a question paper PDF/image and receiving an AI-generated draft rubric.
  - Confirming a draft rubric (with teacher edits) to create a Task + Rubric in one step.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import Task, TaskRubric, User
from app.schemas.common import ApiResponse
from app.schemas.rubric import RubricStep, TaskRubricCreate, TaskResponse, TaskRubricResponse
from app.services.ingestion import validate_file, upload_file
from app.services.question_paper import process_question_paper
from app.services.auth_service import get_current_user

router = APIRouter(prefix="/question-papers", tags=["Question Papers"])


# ── Request/Response Schemas ──

class DraftRubricStep(BaseModel):
    """A single step from the AI-generated draft rubric."""
    step_num: int
    description: str
    marks: int = 0
    step_type: str = "statement"
    component_type: str = "text"
    expected_exprs: list[str] = Field(default_factory=list)
    marking_notes: str = ""
    partial_credit: bool = True
    diagram_relations: list[dict] = Field(default_factory=list)


class DraftRubricResponse(BaseModel):
    """Response from uploading a question paper — contains the AI draft rubric."""
    question_paper_key: str
    extracted_text: str
    draft_rubric: dict  # Contains steps, grading_notes, ai_confidence
    ai_confidence: float


class ConfirmRubricRequest(BaseModel):
    """Request to confirm a draft rubric and create a Task + Rubric."""
    title: str = Field(min_length=1, max_length=500)
    subject: str = Field(min_length=1, max_length=100)
    board: str = Field(description="CBSE | ICSE | State")
    grade_level: str | None = None
    max_marks: int = Field(ge=1)
    description: str | None = None
    question_paper_key: str = Field(description="S3 key from the upload step")
    rubric: TaskRubricCreate = Field(description="Teacher-reviewed rubric (may be edited from draft)")


# ── Endpoints ──

@router.post("/upload")
async def upload_question_paper(
    file: UploadFile = File(...),
    subject: str = Form("General"),
    board: str = Form("CBSE"),
    grade_level: str = Form("Class 12"),
    max_marks: int = Form(100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload a question paper PDF/image → AI extracts questions and generates a draft rubric.

    The teacher should review the generated rubric and use the /confirm endpoint
    to finalize the Task + Rubric creation.
    """
    # Read and validate file
    file_data = await file.read()
    filename = file.filename or "question_paper.pdf"

    is_valid, error_msg = validate_file(filename, len(file_data))
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    # Upload question paper to S3
    content_type = file.content_type or "application/octet-stream"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "pdf"

    # Upload to a dedicated qpapers/ prefix
    from app.services.ingestion import get_s3_client, ensure_bucket_exists
    from app.config import settings
    import uuid as uuid_mod

    s3_client = get_s3_client()
    ensure_bucket_exists(s3_client)
    qpaper_key = f"qpapers/{uuid_mod.uuid4()}.{ext}"
    s3_client.put_object(
        Bucket=settings.S3_BUCKET,
        Key=qpaper_key,
        Body=file_data,
        ContentType=content_type,
    )

    # Determine file type for parsing
    file_type = ext if ext in ("pdf", "png", "jpg", "jpeg") else "pdf"

    # Process question paper: extract text + generate rubric
    try:
        result = process_question_paper(
            file_data=file_data,
            file_type=file_type,
            subject=subject,
            board=board,
            grade_level=grade_level,
            max_marks=max_marks,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question paper: {str(e)}"
        )

    return ApiResponse(data=DraftRubricResponse(
        question_paper_key=qpaper_key,
        extracted_text=result["extracted_text"],
        draft_rubric=result["draft_rubric"],
        ai_confidence=result["ai_confidence"],
    ))


@router.post("/confirm")
async def confirm_question_paper(
    payload: ConfirmRubricRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Confirm a draft rubric (with teacher edits) and create the Task + Rubric in one step.

    This replaces the manual POST /tasks + POST /tasks/{id}/rubrics flow when
    the teacher uses question paper upload.
    """
    # Create the Task
    task = Task(
        title=payload.title,
        subject=payload.subject,
        board=payload.board,
        grade_level=payload.grade_level,
        max_marks=payload.max_marks,
        description=payload.description,
        question_paper_key=payload.question_paper_key,
    )
    db.add(task)
    await db.flush()

    # Create the Rubric
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
        id=str(rubric.id),
        task_id=str(task.id),
        version=rubric.version,
        steps=payload.rubric.steps,
        grading_notes=rubric.grading_notes,
        is_active=True,
        created_at=rubric.created_at.isoformat(),
    )

    response = TaskResponse(
        id=str(task.id),
        title=task.title,
        subject=task.subject,
        board=task.board,
        grade_level=task.grade_level,
        max_marks=task.max_marks,
        description=task.description,
        baseline_run_id=None,
        created_at=task.created_at.isoformat(),
        updated_at=task.updated_at.isoformat(),
        current_rubric=rubric_response,
    )

    return ApiResponse(data=response)

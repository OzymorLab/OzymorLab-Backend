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
from app.schemas.rubric import QuestionPaperResponse, RubricDraftResponse, RubricConfirmRequest
from app.services.auth_service import get_current_user, require_role

router = APIRouter(
    prefix="/question-papers", 
    tags=["Question Papers"],
    dependencies=[Depends(require_role(["teacher", "admin", "hod", "principal"]))]
)


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
    # Phase 4 additions
    exam_cycle_id: str | None = None
    paper_set: str | None = None


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

    # Upload question paper to Supabase Storage in 'qpapers' folder
    content_type = file.content_type or "application/octet-stream"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "pdf"
    from app.services.ingestion import upload_file

    try:
        qpaper_key = upload_file(file_data, filename, content_type, folder="qpapers")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload question paper to Supabase Storage: {str(e)}"
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
    import uuid

    # Create the Task
    task = Task(
        title=payload.title,
        subject=payload.subject,
        board=payload.board,
        grade_level=payload.grade_level,
        max_marks=payload.max_marks,
        description=payload.description,
        question_paper_key=payload.question_paper_key,
        # Phase 4 linkages
        exam_cycle_id=uuid.UUID(payload.exam_cycle_id) if payload.exam_cycle_id else None,
        paper_set=payload.paper_set,
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


# ── Rubric Approval Workflow ──

@router.post("/{task_id}/rubric/submit-for-approval",
             dependencies=[Depends(require_role(["teacher", "admin"]))])
async def submit_rubric_for_approval(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Teacher submits rubric for HOD review.
    Transitions approval_status: DRAFT → PENDING_APPROVAL.
    Also handles REJECTED → PENDING_APPROVAL (resubmission after edits).
    """
    from app.schemas.operations import RubricApprovalResponse

    # Fetch active rubric for this task
    result = await db.execute(
        select(TaskRubric).filter_by(task_id=task_id, is_active=True)
        .order_by(TaskRubric.created_at.desc()).limit(1)
    )
    rubric = result.scalar_one_or_none()
    if not rubric:
        raise HTTPException(status_code=404, detail="No active rubric found for this task.")

    if rubric.approval_status not in ("DRAFT", "REJECTED"):
        raise HTTPException(
            status_code=400,
            detail=f"Rubric is already '{rubric.approval_status}'. Only DRAFT or REJECTED rubrics can be submitted.",
        )

    rubric.approval_status = "PENDING_APPROVAL"
    rubric.rejection_notes = None  # Clear previous rejection notes on resubmission
    await db.flush()

    return ApiResponse(data=RubricApprovalResponse(
        rubric_id=str(rubric.id),
        task_id=str(rubric.task_id),
        approval_status=rubric.approval_status,
    ))


@router.post("/{task_id}/rubric/approve",
             dependencies=[Depends(require_role(["hod", "principal", "admin"]))])
async def approve_rubric(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    HOD/Principal approves a rubric.
    Transitions: PENDING_APPROVAL → APPROVED.
    Idempotent: calling on an already-APPROVED rubric returns 200.
    """
    from datetime import datetime, timezone
    from app.schemas.operations import RubricApprovalResponse

    result = await db.execute(
        select(TaskRubric).filter_by(task_id=task_id, is_active=True)
        .order_by(TaskRubric.created_at.desc()).limit(1)
    )
    rubric = result.scalar_one_or_none()
    if not rubric:
        raise HTTPException(status_code=404, detail="No active rubric found for this task.")

    # Idempotent: already approved
    if rubric.approval_status == "APPROVED":
        return ApiResponse(data=RubricApprovalResponse(
            rubric_id=str(rubric.id),
            task_id=str(rubric.task_id),
            approval_status=rubric.approval_status,
            approved_by=str(rubric.approved_by) if rubric.approved_by else None,
            approved_at=rubric.approved_at.isoformat() if rubric.approved_at else None,
        ))

    if rubric.approval_status != "PENDING_APPROVAL":
        raise HTTPException(
            status_code=400,
            detail=f"Rubric is '{rubric.approval_status}'. Only PENDING_APPROVAL rubrics can be approved.",
        )

    rubric.approval_status = "APPROVED"
    rubric.approved_by = current_user.id
    rubric.approved_at = datetime.now(timezone.utc)
    await db.flush()

    return ApiResponse(data=RubricApprovalResponse(
        rubric_id=str(rubric.id),
        task_id=str(rubric.task_id),
        approval_status=rubric.approval_status,
        approved_by=str(rubric.approved_by),
        approved_at=rubric.approved_at.isoformat(),
    ))


@router.post("/{task_id}/rubric/reject",
             dependencies=[Depends(require_role(["hod", "principal", "admin"]))])
async def reject_rubric(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    notes: str = "",
):
    """
    HOD/Principal rejects a rubric with feedback notes.
    Transitions: PENDING_APPROVAL → REJECTED.
    Teacher can then edit and resubmit.
    """
    from app.schemas.operations import RubricApprovalResponse

    result = await db.execute(
        select(TaskRubric).filter_by(task_id=task_id, is_active=True)
        .order_by(TaskRubric.created_at.desc()).limit(1)
    )
    rubric = result.scalar_one_or_none()
    if not rubric:
        raise HTTPException(status_code=404, detail="No active rubric found for this task.")

    if rubric.approval_status != "PENDING_APPROVAL":
        raise HTTPException(
            status_code=400,
            detail=f"Rubric is '{rubric.approval_status}'. Only PENDING_APPROVAL rubrics can be rejected.",
        )

    rubric.approval_status = "REJECTED"
    rubric.rejection_notes = notes or "No notes provided."
    await db.flush()

    return ApiResponse(data=RubricApprovalResponse(
        rubric_id=str(rubric.id),
        task_id=str(rubric.task_id),
        approval_status=rubric.approval_status,
        rejection_notes=rubric.rejection_notes,
    ))


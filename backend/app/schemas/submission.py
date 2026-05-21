"""
Submission schemas — request/response models for student submissions.
"""
from pydantic import BaseModel, Field
from typing import Literal


class ParsedStep(BaseModel):
    """A single parsed step from a student's answer."""
    step_num: int
    text: str
    equations: list[str] = Field(default_factory=list)
    step_type: Literal["statement", "derivation", "result", "diagram"] = "statement"


class ParsedContent(BaseModel):
    """Structured parsed content from a submission."""
    steps: list[ParsedStep] = Field(default_factory=list)
    detected_language: str = "english"
    has_diagrams: bool = False
    parse_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SubmissionCreate(BaseModel):
    """Metadata sent alongside file upload."""
    task_id: str
    student_id: str


class SubmissionResponse(BaseModel):
    """Full submission response with parsed content."""
    id: str
    task_id: str
    student_id: str | None = None
    file_name: str | None
    file_type: str | None
    status: str
    raw_text: str | None = None
    parsed_content: ParsedContent | None = None
    error_message: str | None = None
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class SubmissionListResponse(BaseModel):
    """Minimal submission info for list views."""
    id: str
    task_id: str
    student_id: str | None = None
    file_name: str | None
    status: str
    created_at: str

    model_config = {"from_attributes": True}


class SubmissionUploadResponse(BaseModel):
    """Response after successful file upload + queue."""
    submission_id: str
    status: str = "PENDING"
    message: str = "Submission queued for parsing"

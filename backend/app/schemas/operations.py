"""
Pydantic schemas for Exam Cycle and School Admin operations.
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


# ── Exam Cycle Schemas ──

class ExamCycleCreate(BaseModel):
    """Request to create a new exam cycle."""
    name: str = Field(min_length=1, max_length=255, description="e.g., 'Mid-Term Oct 2026'")
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class ExamCycleUpdate(BaseModel):
    """Request to update an exam cycle."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: Optional[str] = Field(None, pattern="^(ACTIVE|COMPLETED|ARCHIVED)$")


class ExamCycleResponse(BaseModel):
    """Response for a single exam cycle."""
    id: str
    school_id: Optional[str] = None
    name: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    status: str
    created_by: str
    created_at: str
    task_count: int = 0


class ExamCycleDetailResponse(ExamCycleResponse):
    """Detailed exam cycle response with linked tasks."""
    tasks: List[dict] = Field(default_factory=list)


# ── School Admin Schemas ──

class BulkInviteRequest(BaseModel):
    """Request to bulk invite teachers to a school."""
    emails: List[str] = Field(min_length=1, max_length=50, description="List of teacher email addresses")
    role: str = Field(default="teacher", pattern="^(teacher|hod|admin)$")


class BulkInviteResponse(BaseModel):
    """Response from bulk invite."""
    invited: int
    skipped: int
    errors: List[dict] = Field(default_factory=list)


class StudentImportRow(BaseModel):
    """A single row from a student CSV import."""
    roll_number: str
    name: str
    class_name: str
    section_name: str


class StudentImportResponse(BaseModel):
    """Response from CSV student import."""
    created: int
    updated: int
    errors: List[dict] = Field(default_factory=list)
    total_rows: int


# ── Rubric Approval Schemas ──

class RubricApprovalResponse(BaseModel):
    """Response from rubric approval/rejection."""
    rubric_id: str
    task_id: str
    approval_status: str
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    rejection_notes: Optional[str] = None


class RubricRejectRequest(BaseModel):
    """Request to reject a rubric with feedback notes."""
    notes: str = Field(min_length=1, max_length=2000, description="Reason for rejection / feedback for the teacher")


# ── Report Schemas ──

class TaskSummaryResponse(BaseModel):
    """Aggregate statistics for a task's grading results."""
    task_id: str
    task_title: str
    total_submissions: int
    graded_count: int
    mean_score: float
    median_score: float
    min_score: float
    max_score: float
    score_distribution: List[int] = Field(default_factory=list, description="Histogram bins [0-10%, 10-20%, ..., 90-100%]")


class SchoolDashboardResponse(BaseModel):
    """School-wide aggregate dashboard data."""
    school_id: str
    school_name: str
    total_teachers: int
    total_students: int
    total_exam_cycles: int
    total_tasks: int
    total_submissions: int
    total_graded: int

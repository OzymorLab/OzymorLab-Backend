"""
Grade schemas — response models for grading results, component breakdowns, and run statistics.
"""
from pydantic import BaseModel, Field
from typing import Literal


class StepGradeResult(BaseModel):
    """Per-step grading result."""
    step_num: int
    marks_awarded: int
    max_marks: int
    grade_distribution: list[float] = Field(description="Length = max_marks + 1")
    justification: str
    error_type: str | None = None  # null | algebraic_error | missing_step | wrong_formula | presentation
    sympy_valid: bool | None = None  # None = SymPy could not parse
    sympy_error: str | None = None

class ComponentGradeResult(BaseModel):
    """Per-component evaluation result from the multimodal pipeline."""
    type: Literal["text", "diagram", "labels", "reasoning"]
    marks_awarded: int
    max_marks: int
    confidence: float = Field(ge=0.0, le=1.0)
    grade_distribution: list[float] = Field(default_factory=list)
    justification: str = ""
    # Optional component-specific details
    step_grades: list[StepGradeResult] | None = None  # for text/reasoning
    label_details: list[dict] | None = None  # for labels


class GradeResultResponse(BaseModel):
    """Full grade result with component-level trace and review status."""
    id: str
    submission_id: str
    grading_run_id: str
    grade: int
    max_grade: int
    grade_distribution: list[float] = Field(description="Length = max_grade + 1, sums to 1.0")
    confidence: float = Field(ge=0.0, le=1.0)
    step_grades: list[StepGradeResult]
    component_grades: list[ComponentGradeResult] | None = None
    justification: str | None
    model_used: str
    graded_at: str
    latency_ms: int | None
    # Human review fields
    review_status: str = "AUTO_GRADED"  # AUTO_GRADED | NEEDS_REVIEW | REVIEWED | OVERRIDDEN
    review_reasons: list[str] | None = None
    flagged_components: list[str] | None = None
    review_notes: str | None = None
    reviewed_by: str | None = None

    model_config = {"from_attributes": True}


class GradingRunCreate(BaseModel):
    """Schema for creating a new grading run."""
    task_id: str
    rubric_version: str | None = Field(default=None, description="If null, uses latest active rubric")
    model: str | None = Field(default=None, description="Override default Gemini model")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    description: str | None = None


class GradingRunResponse(BaseModel):
    """Grading run status and progress."""
    id: str
    task_id: str
    rubric_version: str
    model: str
    temperature: float
    description: str | None
    status: str  # CREATED | RUNNING | COMPLETED | FAILED
    total_submissions: int
    graded_count: int
    failed_count: int
    created_at: str
    completed_at: str | None

    model_config = {"from_attributes": True}


class RunStatistics(BaseModel):
    """Aggregated statistics for a grading run."""
    run_id: str
    submission_count: int
    graded_count: int
    failed_count: int
    mean_grade: float
    median_grade: float
    p25_grade: float
    p75_grade: float
    mean_confidence: float
    mean_latency_ms: float
    aggregate_distribution: list[float]
    most_common_error: str | None

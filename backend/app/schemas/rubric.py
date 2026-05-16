"""
Rubric schemas — request/response models for tasks and rubrics.
"""
from pydantic import BaseModel, Field
from typing import Literal


class RubricStep(BaseModel):
    """A single step in a rubric — defines expectations for grading."""
    step_num: int
    description: str
    marks: int = Field(ge=0, description="Max marks for this step")
    expected_exprs: list[str] = Field(default_factory=list, description="SymPy-parseable expected equations")
    marking_notes: str = ""
    step_type: Literal["statement", "derivation", "result", "diagram"] = "statement"
    partial_credit: bool = True
    component_type: Literal["text", "diagram", "labels", "reasoning"] | None = Field(
        default=None,
        description=(
            "Explicitly tags which evaluation pipeline should handle this step. "
            "If set on all steps, the system uses teacher-defined decomposition. "
            "If not set, the system auto-decomposes using LLM or step_type fallback."
        ),
    )
    diagram_relations: list[dict] = Field(
        default_factory=list,
        description=(
            "For diagram steps only. Expected label→region mappings used by the DEIS "
            "Diagram-marker for graph isomorphism scoring. "
            "Example: [{'label': 'aorta', 'region': 'region_0'}, {'label': 'left ventricle', 'region': 'region_1'}]"
        ),
    )


class TaskRubricCreate(BaseModel):
    """Schema for creating/uploading a new rubric version."""
    version: str = Field(description="Semver e.g. '1.0.0'")
    steps: list[RubricStep]
    grading_notes: str = ""


class TaskRubricResponse(BaseModel):
    """Rubric response with full step details."""
    id: str
    task_id: str
    version: str
    steps: list[RubricStep]
    grading_notes: str | None
    is_active: bool
    created_at: str

    model_config = {"from_attributes": True}


class TaskCreate(BaseModel):
    """Schema for creating a new assessment task."""
    title: str = Field(min_length=1, max_length=500)
    subject: str = Field(min_length=1, max_length=100)
    board: str = Field(description="CBSE | ICSE | State")
    grade_level: str | None = Field(default=None, description="e.g. 'Class 12'")
    max_marks: int = Field(ge=1)
    description: str | None = None
    rubric: TaskRubricCreate | None = Field(default=None, description="Optional initial rubric")


class TaskResponse(BaseModel):
    """Full task response with current rubric."""
    id: str
    title: str
    subject: str
    board: str
    grade_level: str | None
    max_marks: int
    description: str | None
    baseline_run_id: str | None
    created_at: str
    updated_at: str
    current_rubric: TaskRubricResponse | None = None

    model_config = {"from_attributes": True}


class TaskListResponse(BaseModel):
    """Minimal task info for list views."""
    id: str
    title: str
    subject: str
    board: str
    max_marks: int
    created_at: str

    model_config = {"from_attributes": True}

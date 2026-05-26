"""
SQLAlchemy ORM models for the AIOS database schema.
All tables from the technical proposal are defined here.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship

from app.db.session import Base


def utcnow():
    return datetime.now(timezone.utc)


import base64

import hashlib
from cryptography.fernet import Fernet
from app.config import settings

def get_fernet() -> Fernet:
    secret = settings.JWT_SECRET_KEY or "edexia-secret-change-me-in-production"
    key_bytes = hashlib.sha256(secret.encode()).digest()
    fernet_key = base64.urlsafe_b64encode(key_bytes)
    return Fernet(fernet_key)

def encrypt_key(raw_key: str) -> str:
    if not raw_key:
        return None
    try:
        f = get_fernet()
        return f.encrypt(raw_key.encode()).decode()
    except Exception:
        return raw_key

def decrypt_key(encrypted_key: str) -> str:
    if not encrypted_key:
        return None
    try:
        f = get_fernet()
        return f.decrypt(encrypted_key.encode()).decode()
    except Exception:
        return encrypted_key


class School(Base):
    """A school entity that holds users (teachers/admins) and classes."""
    __tablename__ = "schools"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    
    users = relationship("User", back_populates="school", cascade="all, delete-orphan")
    classes = relationship("SchoolClass", back_populates="school", cascade="all, delete-orphan")
    exam_cycles = relationship("ExamCycle", back_populates="school", cascade="all, delete-orphan")


class ExamCycle(Base):
    """Groups multiple subject papers into a single exam event (e.g., 'Mid-Term Oct 2026')."""
    __tablename__ = "exam_cycles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    school_id = Column(UUID(as_uuid=True), ForeignKey("schools.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)  # e.g., "Mid-Term Oct 2026"
    start_date = Column(DateTime(timezone=True), nullable=True)
    end_date = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), nullable=False, default="ACTIVE")  # ACTIVE | COMPLETED | ARCHIVED
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    school = relationship("School", back_populates="exam_cycles")
    tasks = relationship("Task", back_populates="exam_cycle")
    creator = relationship("User", foreign_keys=[created_by])

    __table_args__ = (
        Index("idx_exam_cycles_school_id", "school_id"),
    )


class SchoolClass(Base):
    """A class standard within a school (e.g., 'Grade 10')."""
    __tablename__ = "classes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    school_id = Column(UUID(as_uuid=True), ForeignKey("schools.id"), nullable=False, index=True)
    name = Column(String(100), nullable=False)  # e.g., "X", "XII"
    created_at = Column(DateTime(timezone=True), default=utcnow)
    
    school = relationship("School", back_populates="classes")
    sections = relationship("Section", back_populates="school_class", cascade="all, delete-orphan")


class Section(Base):
    """A specific section within a class (e.g., 'A', 'B')."""
    __tablename__ = "sections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    class_id = Column(UUID(as_uuid=True), ForeignKey("classes.id"), nullable=False, index=True)
    name = Column(String(100), nullable=False)  # e.g., "A", "Science"
    created_at = Column(DateTime(timezone=True), default=utcnow)
    
    school_class = relationship("SchoolClass", back_populates="sections")
    students = relationship("Student", back_populates="section", cascade="all, delete-orphan")


class Student(Base):
    """A student belonging to a specific section."""
    __tablename__ = "students"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    section_id = Column(UUID(as_uuid=True), ForeignKey("sections.id"), nullable=False, index=True)
    roll_number = Column(String(50), nullable=False)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    
    section = relationship("Section", back_populates="students")
    submissions = relationship("Submission", back_populates="student")



class User(Base):
    """User account — teachers, evaluators, admins."""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="teacher")
    # teacher | admin | student | hod | principal
    school_id = Column(UUID(as_uuid=True), ForeignKey("schools.id"), nullable=True, index=True)
    
    school = relationship("School", back_populates="users")
    
    # Map the column to a private attribute to enable automatic encryption/decryption
    _gemini_api_key = Column("gemini_api_key", Text, nullable=True)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    @property
    def gemini_api_key(self) -> str:
        """Transparently decrypt the API key when read."""
        if not self._gemini_api_key:
            return None
        return decrypt_key(self._gemini_api_key)

    @gemini_api_key.setter
    def gemini_api_key(self, value: str):
        """Transparently encrypt the API key when written."""
        if not value:
            self._gemini_api_key = None
        else:
            self._gemini_api_key = encrypt_key(value)

    __table_args__ = (
        Index("idx_users_email", "email", unique=True),
        Index("idx_users_role", "role"),
    )



class Task(Base):
    """Assessment task definition (e.g., a specific Physics question)."""
    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    subject = Column(String(100), nullable=False)
    board = Column(String(50), nullable=False)  # CBSE | ICSE | State
    grade_level = Column(String(20), nullable=True)  # e.g., "Class 12"
    max_marks = Column(Integer, nullable=False)
    description = Column(Text, nullable=True)
    question_paper_key = Column(Text, nullable=True)  # S3 object key for uploaded question paper
    baseline_run_id = Column(UUID(as_uuid=True), nullable=True)  # designated drift baseline
    # Phase 4: Exam cycle and paper set support
    exam_cycle_id = Column(UUID(as_uuid=True), ForeignKey("exam_cycles.id"), nullable=True, index=True)
    paper_set = Column(String(10), nullable=True)  # A, B, C, or NULL for standalone tasks
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    rubrics = relationship("TaskRubric", back_populates="task", lazy="selectin")
    submissions = relationship("Submission", back_populates="task", lazy="selectin")
    grading_runs = relationship("GradingRun", back_populates="task", lazy="selectin")
    drift_reports = relationship("DriftReport", back_populates="task", lazy="selectin")
    exam_cycle = relationship("ExamCycle", back_populates="tasks")


class TaskRubric(Base):
    """Versioned rubric for a task — the source of truth for grading."""
    __tablename__ = "task_rubrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=False)
    version = Column(String(20), nullable=False)  # semver e.g., "1.0.0"
    rubric_json = Column(JSONB, nullable=False)  # full rubric with steps
    grading_notes = Column(Text, nullable=True)  # board-level guidance
    is_active = Column(Boolean, default=True)
    # Phase 4: Rubric approval workflow
    approval_status = Column(String(20), nullable=False, default="DRAFT")  # DRAFT | PENDING_APPROVAL | APPROVED | REJECTED
    approved_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    approved_at = Column(DateTime(timezone=True), nullable=True)
    rejection_notes = Column(Text, nullable=True)  # HOD's feedback when rejecting
    created_at = Column(DateTime(timezone=True), default=utcnow)

    # Relationships
    task = relationship("Task", back_populates="rubrics")
    approver = relationship("User", foreign_keys=[approved_by])

    __table_args__ = (
        Index("idx_task_rubrics_task_version", "task_id", "version", unique=True),
        Index("idx_task_rubrics_approval_status", "approval_status"),
    )


class Submission(Base):
    """Student submission — a file uploaded for grading."""
    __tablename__ = "submissions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=False)
    student_id = Column(UUID(as_uuid=True), ForeignKey("students.id"), nullable=True, index=True)
    file_key = Column(Text, nullable=False)  # S3 object key
    file_name = Column(String(500), nullable=True)
    file_type = Column(String(20), nullable=True)  # pdf, png, jpg, jpeg
    raw_text = Column(Text, nullable=True)  # extracted text
    parsed_content = Column(JSONB, nullable=True)  # structured answer steps
    status = Column(String(20), nullable=False, default="PENDING")
    # PENDING | PARSING | PARSED | GRADING | GRADED | FAILED
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    task = relationship("Task", back_populates="submissions")
    grade_results = relationship("GradeResult", back_populates="submission", lazy="selectin")
    student = relationship("Student", back_populates="submissions")
    steps = relationship("SubmissionStep", back_populates="submission", cascade="all, delete-orphan", order_by="SubmissionStep.step_num")

    # Multimodal evaluation
    question_decomposition = Column(JSONB, nullable=True)  # cached decomposition result

    __table_args__ = (
        Index("idx_submissions_task_id", "task_id"),
        Index("idx_submissions_status", "status"),
    )


class SubmissionStep(Base):
    """A structured step within a student submission, containing OCR and grading traces."""
    __tablename__ = "submission_steps"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    submission_id = Column(UUID(as_uuid=True), ForeignKey("submissions.id"), nullable=False, index=True)
    step_num = Column(Integer, nullable=False)
    step_type = Column(String(100), nullable=True)  # e.g., "Substitution Setup", "Integration"
    text = Column(Text, nullable=True)             # OCR description text
    latex = Column(Text, nullable=True)            # LaTeX equation
    sympy_valid = Column(Boolean, nullable=True)    # computational validity (True, False, or Null)
    justification = Column(Text, nullable=True)    # AI grading rationale
    marks_awarded = Column(Float, nullable=False, default=0.0)
    max_marks = Column(Float, nullable=False, default=0.0)
    error_type = Column(String(100), nullable=True) # Sign Error, Arithmetic Flub
    bounding_box = Column(JSONB, nullable=True)     # OCR box: {x, y, w, h}
    created_at = Column(DateTime(timezone=True), default=utcnow)

    submission = relationship("Submission", back_populates="steps")


class GradingRun(Base):
    """A batch grading run — one run grades all submissions for a task."""
    __tablename__ = "grading_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=False)
    rubric_version = Column(String(20), nullable=False)
    model = Column(String(100), nullable=False)
    temperature = Column(Float, nullable=False, default=0.0)
    description = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="CREATED")
    # CREATED | RUNNING | COMPLETED | FAILED
    total_submissions = Column(Integer, default=0)
    graded_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)  # who triggered this run (for BYOK key lookup)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    task = relationship("Task", back_populates="grading_runs")
    grade_results = relationship("GradeResult", back_populates="grading_run", lazy="selectin")

    __table_args__ = (
        Index("idx_grading_runs_task_id", "task_id"),
    )


class GradeResult(Base):
    """Per-submission grade output from a grading run."""
    __tablename__ = "grade_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    submission_id = Column(UUID(as_uuid=True), ForeignKey("submissions.id"), nullable=False)
    grading_run_id = Column(UUID(as_uuid=True), ForeignKey("grading_runs.id"), nullable=False)
    grade = Column(Integer, nullable=False)
    max_grade = Column(Integer, nullable=False)
    grade_distribution = Column(JSONB, nullable=False)  # float array, sums to 1.0
    confidence = Column(Float, nullable=True)  # scalar summary of distribution sharpness
    step_grades = Column(JSONB, nullable=False)  # array of per-step results
    justification = Column(Text, nullable=True)
    llm_call_ids = Column(ARRAY(String), nullable=True)  # trace to raw LLM calls
    model_used = Column(String(100), nullable=False)
    graded_at = Column(DateTime(timezone=True), default=utcnow)
    latency_ms = Column(Integer, nullable=True)

    # Multimodal component evaluation
    component_grades = Column(JSONB, nullable=True)  # per-component breakdown (text, diagram, labels, reasoning)
    review_status = Column(String(20), nullable=False, default="AUTO_GRADED")
    # AUTO_GRADED | NEEDS_REVIEW | REVIEWED | OVERRIDDEN
    review_reasons = Column(JSONB, nullable=True)  # why human review is needed
    flagged_components = Column(JSONB, nullable=True)  # list of flagged component types
    review_notes = Column(Text, nullable=True)  # teacher's moderation notes
    reviewed_by = Column(String(255), nullable=True)  # teacher/evaluator ID
    reviewed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    submission = relationship("Submission", back_populates="grade_results")
    grading_run = relationship("GradingRun", back_populates="grade_results")

    __table_args__ = (
        Index("idx_grade_results_run_id", "grading_run_id"),
        Index("idx_grade_results_submission_id", "submission_id"),
        Index("idx_grade_results_review_status", "review_status"),
    )


class DriftReport(Base):
    """Drift analysis comparing a grading run to the baseline."""
    __tablename__ = "drift_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=False)
    current_run_id = Column(UUID(as_uuid=True), ForeignKey("grading_runs.id"), nullable=False)
    baseline_run_id = Column(UUID(as_uuid=True), ForeignKey("grading_runs.id"), nullable=False)
    kl_divergence = Column(Float, nullable=False)
    mean_shift = Column(Float, nullable=False)
    entropy_current = Column(Float, nullable=False)
    entropy_baseline = Column(Float, nullable=False)
    drift_detected = Column(Boolean, nullable=False)
    severity = Column(String(10), nullable=True)  # LOW | MEDIUM | HIGH
    details = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    # Relationships
    task = relationship("Task", back_populates="drift_reports")

    __table_args__ = (
        Index("idx_drift_reports_task_id", "task_id"),
    )


class GradingAlert(Base):
    """System alerts generated by the observability layer."""
    __tablename__ = "grading_alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("grading_runs.id"), nullable=True)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=True)
    alert_type = Column(String(30), nullable=False)
    # DRIFT | LATENCY | FAILURE_RATE | UNCERTAINTY
    severity = Column(String(10), nullable=False)  # LOW | MEDIUM | HIGH
    message = Column(Text, nullable=False)
    metadata_json = Column(JSONB, nullable=True)
    resolved = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        Index("idx_grading_alerts_task_id", "task_id"),
        Index("idx_grading_alerts_resolved", "resolved"),
    )


class LLMCallLog(Base):
    """Raw LLM prompt/response log for traceability."""
    __tablename__ = "llm_call_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    submission_id = Column(UUID(as_uuid=True), ForeignKey("submissions.id"), nullable=True)
    grading_run_id = Column(UUID(as_uuid=True), ForeignKey("grading_runs.id"), nullable=True)
    call_type = Column(String(50), nullable=False)  # alignment | step_grading | parsing
    model = Column(String(100), nullable=False)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=True)
    tokens_in = Column(Integer, nullable=True)
    tokens_out = Column(Integer, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        Index("idx_llm_calls_submission_id", "submission_id"),
        Index("idx_llm_calls_run_id", "grading_run_id"),
    )


class IdempotencyKey(Base):
    """Tracks idempotency signatures to guarantee safe API retries and transaction stability."""
    __tablename__ = "idempotency_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_hash = Column(String(64), unique=True, nullable=False, index=True)
    status = Column(String(20), default="PROCESSING")  # PROCESSING | SUCCESS | FAILED
    response_code = Column(Integer, nullable=True)
    response_body = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)


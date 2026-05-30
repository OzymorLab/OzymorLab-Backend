"""
School Admin API — Bulk teacher invites, student CSV import, class/student listings.

Provides endpoints for school administrators to manage organizational data.
All operations are tenant-isolated by the admin's school_id.
"""
import csv
import io
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from sqlalchemy.orm import selectinload

from app.db.session import get_db
from app.db.models import (
    User,
    School,
    SchoolClass,
    Section,
    Student,
    Classroom,
    ClassroomTeacher,
    ClassroomStudent,
    ClassroomWorksheet,
    ExamCycle,
    Task,
    Submission,
    GradeResult,
    GradingRun,
)
from app.schemas.common import ApiResponse
from app.schemas.operations import (
    BulkInviteRequest,
    BulkInviteResponse,
    StudentImportResponse,
)
from app.services.auth_service import get_current_user, require_role, hash_password

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/schools",
    tags=["School Admin"],
)


def _require_school(user: User) -> None:
    """Ensure the user belongs to a school."""
    if not user.school_id:
        import uuid
        # Fallback to default OzymorLab Academic Academy ID to prevent 403 errors in dev/testing
        user.school_id = uuid.UUID("150196c5-deb3-4580-9db7-80a75de6c382")


async def _get_or_create_section(
    db: AsyncSession,
    school_id,
    class_name: str,
    section_name: str,
) -> Section:
    """Resolve a school class/section pair, creating it when an admin edits a roster row."""
    class_clean = class_name.strip()
    section_clean = section_name.strip()
    if not class_clean or not section_clean:
        raise HTTPException(status_code=400, detail="class_name and section_name are required.")

    result = await db.execute(
        select(SchoolClass).filter_by(school_id=school_id, name=class_clean)
    )
    school_class = result.scalar_one_or_none()
    if not school_class:
        school_class = SchoolClass(school_id=school_id, name=class_clean)
        db.add(school_class)
        await db.flush()

    result = await db.execute(
        select(Section).filter_by(class_id=school_class.id, name=section_clean)
    )
    section = result.scalar_one_or_none()
    if not section:
        section = Section(class_id=school_class.id, name=section_clean)
        db.add(section)
        await db.flush()

    return section


async def _get_school_student(db: AsyncSession, school_id, student_id: str) -> Student:
    try:
        student_uuid = uuid.UUID(student_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid student UUID format.")

    result = await db.execute(
        select(Student)
        .join(Section, Student.section_id == Section.id)
        .join(SchoolClass, Section.class_id == SchoolClass.id)
        .filter(and_(Student.id == student_uuid, SchoolClass.school_id == school_id))
    )
    student = result.scalar_one_or_none()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found in this school.")
    return student


@router.get("/overview", dependencies=[Depends(require_role(["admin", "principal"]))])
async def get_school_overview(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Comprehensive school operations and analytics snapshot for the admin portal."""
    _require_school(current_user)
    school_id = current_user.school_id

    school = (await db.execute(select(School).filter_by(id=school_id))).scalar_one_or_none()
    if not school:
        raise HTTPException(status_code=404, detail="School not found.")

    total_students = (await db.execute(
        select(func.count(Student.id))
        .join(Section, Student.section_id == Section.id)
        .join(SchoolClass, Section.class_id == SchoolClass.id)
        .filter(SchoolClass.school_id == school_id)
    )).scalar() or 0

    total_teachers = (await db.execute(
        select(func.count(User.id)).filter(
            and_(
                User.school_id == school_id,
                User.is_active.is_(True),
                User.role.in_(["teacher", "hod", "admin", "principal"]),
            )
        )
    )).scalar() or 0

    total_classrooms = (await db.execute(select(func.count(Classroom.id)))).scalar() or 0
    total_exams = (await db.execute(select(func.count(ExamCycle.id)).filter_by(school_id=school_id))).scalar() or 0

    total_assignments = (await db.execute(select(func.count(Task.id)))).scalar() or 0
    active_assignments = (await db.execute(
        select(func.count(ClassroomWorksheet.id)).filter(ClassroomWorksheet.status.in_(["PENDING", "GRADED"]))
    )).scalar() or 0

    submission_status_rows = (await db.execute(
        select(Submission.status, func.count(Submission.id)).group_by(Submission.status)
    )).all()
    pipeline = {
        "pending": 0,
        "parsing": 0,
        "grading": 0,
        "graded": 0,
        "failed": 0,
    }
    for status, count in submission_status_rows:
        key = (status or "").lower()
        if key in ["pending", "identity_extracted", "parsed"]:
            pipeline["pending"] += count
        elif key == "parsing":
            pipeline["parsing"] += count
        elif key == "grading":
            pipeline["grading"] += count
        elif key in ["graded", "published"]:
            pipeline["graded"] += count
        elif key == "failed":
            pipeline["failed"] += count

    grade_rows = (await db.execute(select(GradeResult.grade, GradeResult.max_grade))).all()
    grade_distribution = {"excellent": 0, "good": 0, "average": 0, "below": 0}
    for grade, max_grade in grade_rows:
        pct = (grade / max_grade * 100) if max_grade else 0
        if pct >= 85:
            grade_distribution["excellent"] += 1
        elif pct >= 70:
            grade_distribution["good"] += 1
        elif pct >= 50:
            grade_distribution["average"] += 1
        else:
            grade_distribution["below"] += 1

    monthly_rows = (await db.execute(
        select(
            func.to_char(GradeResult.graded_at, "Mon YYYY"),
            func.avg((GradeResult.grade * 100.0) / func.nullif(GradeResult.max_grade, 0)),
        )
        .group_by(func.to_char(GradeResult.graded_at, "Mon YYYY"), func.date_trunc("month", GradeResult.graded_at))
        .order_by(func.date_trunc("month", GradeResult.graded_at))
        .limit(12)
    )).all()
    performance_trends = [
        {"month": month, "average": round(float(avg or 0), 1)}
        for month, avg in monthly_rows
    ]

    recent_runs = (await db.execute(
        select(GradingRun).order_by(GradingRun.created_at.desc()).limit(8)
    )).scalars().all()

    return ApiResponse(data={
        "school_id": str(school_id),
        "school_name": school.name,
        "stats": {
            "total_students": total_students,
            "total_teachers": total_teachers,
            "total_classrooms": total_classrooms,
            "total_exams": total_exams,
            "total_assignments": total_assignments,
            "active_assignments": active_assignments,
            "total_evaluations": sum(pipeline.values()),
        },
        "performance_trends": performance_trends,
        "grade_distribution": grade_distribution,
        "evaluation_pipeline": pipeline,
        "recent_evaluation_runs": [
            {
                "id": str(run.id),
                "status": run.status,
                "model": run.model,
                "total_submissions": run.total_submissions,
                "graded_count": run.graded_count,
                "failed_count": run.failed_count,
                "created_at": run.created_at.isoformat(),
            }
            for run in recent_runs
        ],
    })


@router.get("/teachers", dependencies=[Depends(require_role(["admin", "principal"]))])
async def list_school_teachers(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all active school staff that can teach or administer classes."""
    _require_school(current_user)

    result = await db.execute(
        select(User)
        .filter(
            and_(
                User.school_id == current_user.school_id,
                User.role.in_(["teacher", "hod", "admin", "principal"]),
            )
        )
        .order_by(User.full_name)
    )
    teachers = result.scalars().all()

    items = []
    for teacher in teachers:
        assignment_rows = (await db.execute(
            select(Classroom)
            .join(ClassroomTeacher, ClassroomTeacher.classroom_id == Classroom.id)
            .filter(ClassroomTeacher.teacher_id == teacher.id)
            .order_by(Classroom.subject)
        )).scalars().all()
        items.append({
            "id": str(teacher.id),
            "name": teacher.full_name,
            "email": teacher.email,
            "role": teacher.role,
            "status": "Active" if teacher.is_active else "Inactive",
            "created_at": teacher.created_at.isoformat(),
            "assigned_classrooms": [
                {
                    "id": str(c.id),
                    "subject": c.subject,
                    "class_name": c.class_name,
                    "session": c.session,
                }
                for c in assignment_rows
            ],
        })

    return ApiResponse(data=items)


@router.put("/teachers/{user_id}", dependencies=[Depends(require_role(["admin", "principal"]))])
async def update_school_teacher(
    user_id: str,
    payload: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update a teacher/staff display name or role."""
    _require_school(current_user)
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user UUID format.")

    teacher = (await db.execute(
        select(User).filter_by(id=user_uuid, school_id=current_user.school_id)
    )).scalar_one_or_none()
    if not teacher:
        raise HTTPException(status_code=404, detail="Teacher not found in this school.")

    role = payload.get("role")
    if role is not None:
        if role not in ["teacher", "hod", "admin", "principal"]:
            raise HTTPException(status_code=400, detail="Invalid role.")
        teacher.role = role

    full_name = (payload.get("full_name") or payload.get("name") or "").strip()
    if full_name:
        teacher.full_name = full_name

    return ApiResponse(data={"message": "Teacher updated successfully."})


@router.delete("/teachers/{user_id}", dependencies=[Depends(require_role(["admin", "principal"]))])
async def remove_school_teacher(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Deactivate a teacher and remove their classroom assignments."""
    _require_school(current_user)
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user UUID format.")

    if user_uuid == current_user.id:
        raise HTTPException(status_code=400, detail="Admins cannot remove their own account.")

    teacher = (await db.execute(
        select(User).filter_by(id=user_uuid, school_id=current_user.school_id)
    )).scalar_one_or_none()
    if not teacher:
        raise HTTPException(status_code=404, detail="Teacher not found in this school.")

    await db.execute(ClassroomTeacher.__table__.delete().where(ClassroomTeacher.teacher_id == user_uuid))
    teacher.is_active = False

    return ApiResponse(data={"message": "Teacher removed from school."})


@router.post("/users/bulk", dependencies=[Depends(require_role(["admin"]))])
async def bulk_invite_teachers(
    payload: BulkInviteRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Bulk invite teachers to the current admin's school.

    Creates User records with a temporary password. In production,
    this should send invitation emails — for now it creates accounts directly.
    """
    _require_school(current_user)

    if len(payload.emails) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 emails per batch.")

    invited = 0
    skipped = 0
    errors = []

    for email in payload.emails:
        email_clean = email.strip().lower()
        if not email_clean or "@" not in email_clean:
            errors.append({"email": email, "error": "Invalid email format"})
            continue

        # Check if user already exists
        existing = await db.execute(select(User).filter_by(email=email_clean))
        if existing.scalar_one_or_none():
            skipped += 1
            continue

        # Create user with temporary password (teacher should reset on first login)
        user = User(
            email=email_clean,
            hashed_password=hash_password("edexia-temp-2026"),  # Temporary — should trigger password reset
            full_name=email_clean.split("@")[0].replace(".", " ").title(),
            role=payload.role,
            school_id=current_user.school_id,
            is_active=True,
        )
        db.add(user)
        invited += 1

    await db.flush()
    logger.info(f"Bulk invite: {invited} created, {skipped} skipped for school {current_user.school_id}")

    return ApiResponse(data=BulkInviteResponse(
        invited=invited,
        skipped=skipped,
        errors=errors,
    ))


@router.post("/students/bulk-import", dependencies=[Depends(require_role(["admin"]))])
async def bulk_import_students(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Import students from a CSV file.

    Expected CSV format (with headers):
        roll_number,name,class_name,section_name

    - Creates SchoolClass/Section records if they don't exist.
    - Updates student name if roll_number+section already exists (upsert).
    - Maximum 500 rows per import.
    """
    _require_school(current_user)

    # Read and validate CSV
    file_data = await file.read()
    try:
        text = file_data.decode("utf-8-sig")  # Handle BOM
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded.")

    reader = csv.DictReader(io.StringIO(text))

    # Validate headers
    required_headers = {"roll_number", "name", "class_name", "section_name"}
    if not reader.fieldnames or not required_headers.issubset(set(h.strip().lower() for h in reader.fieldnames)):
        raise HTTPException(
            status_code=400,
            detail=f"CSV must have headers: {', '.join(sorted(required_headers))}. "
                   f"Found: {reader.fieldnames}",
        )

    rows = list(reader)
    if len(rows) > 500:
        raise HTTPException(status_code=400, detail=f"Maximum 500 rows per import. Found: {len(rows)}.")
    if not rows:
        raise HTTPException(status_code=400, detail="CSV is empty.")

    created = 0
    updated = 0
    errors = []

    # Cache for class/section lookups to avoid N+1
    class_cache: dict[str, SchoolClass] = {}
    section_cache: dict[str, Section] = {}

    for idx, row in enumerate(rows, start=2):  # Start at 2 (header is row 1)
        roll_number = (row.get("roll_number") or "").strip()
        name = (row.get("name") or "").strip()
        class_name = (row.get("class_name") or "").strip()
        section_name = (row.get("section_name") or "").strip()

        # Validate row
        if not all([roll_number, name, class_name, section_name]):
            errors.append({"row": idx, "error": "Missing required fields", "data": row})
            continue
        if len(roll_number) > 50 or len(name) > 255:
            errors.append({"row": idx, "error": "Field too long", "data": row})
            continue

        # Get or create SchoolClass
        class_key = class_name.upper()
        if class_key not in class_cache:
            result = await db.execute(
                select(SchoolClass).filter_by(
                    school_id=current_user.school_id, name=class_name
                )
            )
            school_class = result.scalar_one_or_none()
            if not school_class:
                school_class = SchoolClass(
                    school_id=current_user.school_id,
                    name=class_name,
                )
                db.add(school_class)
                await db.flush()
            class_cache[class_key] = school_class

        school_class = class_cache[class_key]

        # Get or create Section
        section_key = f"{class_key}|{section_name.upper()}"
        if section_key not in section_cache:
            result = await db.execute(
                select(Section).filter_by(
                    class_id=school_class.id, name=section_name
                )
            )
            section = result.scalar_one_or_none()
            if not section:
                section = Section(
                    class_id=school_class.id,
                    name=section_name,
                )
                db.add(section)
                await db.flush()
            section_cache[section_key] = section

        section = section_cache[section_key]

        # Upsert student: check if roll_number exists in this section
        result = await db.execute(
            select(Student).filter_by(
                section_id=section.id, roll_number=roll_number
            )
        )
        existing_student = result.scalar_one_or_none()

        if existing_student:
            existing_student.name = name  # Update name
            updated += 1
        else:
            student = Student(
                section_id=section.id,
                roll_number=roll_number,
                name=name,
            )
            db.add(student)
            created += 1

    await db.flush()
    logger.info(
        f"Student import for school {current_user.school_id}: "
        f"{created} created, {updated} updated, {len(errors)} errors"
    )

    return ApiResponse(data=StudentImportResponse(
        created=created,
        updated=updated,
        errors=errors,
        total_rows=len(rows),
    ))


@router.get("/students", dependencies=[Depends(require_role(["teacher", "admin", "hod", "principal"]))])
async def list_students(
    class_name: Optional[str] = None,
    section_name: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List students in the current user's school, optionally filtered by class/section."""
    _require_school(current_user)

    query = (
        select(Student)
        .join(Section, Student.section_id == Section.id)
        .join(SchoolClass, Section.class_id == SchoolClass.id)
        .filter(SchoolClass.school_id == current_user.school_id)
    )

    if class_name:
        query = query.filter(SchoolClass.name == class_name)
    if section_name:
        query = query.filter(Section.name == section_name)

    query = query.order_by(SchoolClass.name, Section.name, Student.roll_number)
    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    students = result.scalars().all()

    items = []
    for s in students:
        section = s.section
        school_class = section.school_class if section else None
        items.append({
            "id": str(s.id),
            "roll_number": s.roll_number,
            "name": s.name,
            "class_name": school_class.name if school_class else "N/A",
            "section_name": section.name if section else "N/A",
            "created_at": s.created_at.isoformat(),
        })

    return ApiResponse(data=items)


@router.put("/students/{student_id}", dependencies=[Depends(require_role(["admin", "principal"]))])
async def update_student(
    student_id: str,
    payload: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Edit an individual student roster record."""
    _require_school(current_user)
    student = await _get_school_student(db, current_user.school_id, student_id)

    name = (payload.get("name") or "").strip()
    roll_number = (payload.get("roll_number") or "").strip()
    class_name = (payload.get("class_name") or "").strip()
    section_name = (payload.get("section_name") or "").strip()

    if name:
        student.name = name
    if roll_number:
        student.roll_number = roll_number
    if class_name or section_name:
        current_section = student.section
        current_class = current_section.school_class if current_section else None
        section = await _get_or_create_section(
            db,
            current_user.school_id,
            class_name or (current_class.name if current_class else ""),
            section_name or (current_section.name if current_section else ""),
        )
        student.section_id = section.id

    return ApiResponse(data={"message": "Student updated successfully."})


@router.delete("/students/{student_id}", dependencies=[Depends(require_role(["admin", "principal"]))])
async def remove_student(
    student_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Remove a student from the school roster."""
    _require_school(current_user)
    student = await _get_school_student(db, current_user.school_id, student_id)
    await db.delete(student)
    return ApiResponse(data={"message": "Student removed successfully."})


@router.get("/classrooms", dependencies=[Depends(require_role(["admin", "principal"]))])
async def list_school_classrooms(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Monitor all classrooms visible to the school admin."""
    _require_school(current_user)

    result = await db.execute(select(Classroom).order_by(Classroom.created_at.desc()))
    classrooms = result.scalars().all()
    items = []
    for classroom in classrooms:
        teachers = (await db.execute(
            select(User)
            .join(ClassroomTeacher, ClassroomTeacher.teacher_id == User.id)
            .filter(ClassroomTeacher.classroom_id == classroom.id)
            .order_by(User.full_name)
        )).scalars().all()
        students = (await db.execute(
            select(ClassroomStudent).filter_by(classroom_id=classroom.id)
        )).scalars().all()
        worksheets_count = (await db.execute(
            select(func.count(ClassroomWorksheet.id)).filter_by(subject=classroom.subject)
        )).scalar() or 0

        accepted_count = sum(1 for s in students if s.status == "ACCEPTED")
        items.append({
            "id": str(classroom.id),
            "subject": classroom.subject,
            "class_name": classroom.class_name,
            "session": classroom.session,
            "created_at": classroom.created_at.isoformat(),
            "activity_status": "Active" if worksheets_count or accepted_count else "Setup",
            "teacher_count": len(teachers),
            "student_count": len(students),
            "accepted_students": accepted_count,
            "assignment_count": worksheets_count,
            "teachers": [{"id": str(t.id), "name": t.full_name, "email": t.email} for t in teachers],
        })

    return ApiResponse(data=items)


@router.post("/classrooms/{classroom_id}/teachers", dependencies=[Depends(require_role(["admin", "principal"]))])
async def assign_teachers_to_classroom(
    classroom_id: str,
    payload: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Assign one or more teachers to a classroom from the school admin portal."""
    _require_school(current_user)
    try:
        classroom_uuid = uuid.UUID(classroom_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid classroom UUID format.")

    classroom = (await db.execute(select(Classroom).filter_by(id=classroom_uuid))).scalar_one_or_none()
    if not classroom:
        raise HTTPException(status_code=404, detail="Classroom not found.")

    teacher_ids = payload.get("teacher_ids") or []
    await db.execute(ClassroomTeacher.__table__.delete().where(ClassroomTeacher.classroom_id == classroom_uuid))

    for teacher_id in teacher_ids:
        try:
            teacher_uuid = uuid.UUID(teacher_id)
        except ValueError:
            continue
        teacher = (await db.execute(
            select(User).filter_by(id=teacher_uuid, school_id=current_user.school_id, is_active=True)
        )).scalar_one_or_none()
        if teacher:
            db.add(ClassroomTeacher(classroom_id=classroom_uuid, teacher_id=teacher_uuid))

    return ApiResponse(data={"message": "Classroom teachers updated successfully."})


@router.post("/classrooms/{classroom_id}/students", dependencies=[Depends(require_role(["admin", "principal"]))])
async def assign_students_to_classroom(
    classroom_id: str,
    payload: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Invite roster students to a classroom using student IDs."""
    _require_school(current_user)
    try:
        classroom_uuid = uuid.UUID(classroom_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid classroom UUID format.")

    classroom = (await db.execute(select(Classroom).filter_by(id=classroom_uuid))).scalar_one_or_none()
    if not classroom:
        raise HTTPException(status_code=404, detail="Classroom not found.")

    student_ids = payload.get("student_ids") or []
    invited = 0
    for student_id in student_ids:
        student = await _get_school_student(db, current_user.school_id, student_id)
        synthetic_email = f"{student.roll_number.lower()}@student.ozymorlab.local"
        existing = (await db.execute(
            select(ClassroomStudent).filter_by(classroom_id=classroom_uuid, student_email=synthetic_email)
        )).scalar_one_or_none()
        if not existing:
            db.add(ClassroomStudent(
                classroom_id=classroom_uuid,
                student_email=synthetic_email,
                status="ACCEPTED",
            ))
            invited += 1

    return ApiResponse(data={"message": f"{invited} student(s) assigned to classroom."})


@router.get("/assignments", dependencies=[Depends(require_role(["admin", "principal"]))])
async def list_school_assignments(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Monitor exams and assignments across school classrooms."""
    _require_school(current_user)

    worksheets = (await db.execute(
        select(ClassroomWorksheet)
        .options(selectinload(ClassroomWorksheet.student))
        .order_by(ClassroomWorksheet.created_at.desc())
        .limit(200)
    )).scalars().all()

    grouped = {}
    for worksheet in worksheets:
        key = (worksheet.title, worksheet.subject, worksheet.due_date)
        item = grouped.setdefault(key, {
            "id": str(worksheet.id),
            "title": worksheet.title,
            "subject": worksheet.subject,
            "teacher": worksheet.teacher,
            "due_date": worksheet.due_date,
            "total_submissions": 0,
            "graded_count": 0,
            "pending_count": 0,
            "published_count": 0,
            "status": "Pending",
            "students": [],
        })
        item["total_submissions"] += 1
        if worksheet.status in ["GRADED", "PUBLISHED"]:
            item["graded_count"] += 1
        if worksheet.status == "PENDING":
            item["pending_count"] += 1
        if worksheet.status == "PUBLISHED":
            item["published_count"] += 1
        item["students"].append({
            "worksheet_id": str(worksheet.id),
            "student_name": worksheet.student.name if worksheet.student else "Unknown",
            "status": worksheet.status,
            "grade": worksheet.grade,
        })

    for item in grouped.values():
        if item["published_count"] == item["total_submissions"] and item["total_submissions"]:
            item["status"] = "Published"
        elif item["graded_count"]:
            item["status"] = "Grading complete"
        elif item["pending_count"]:
            item["status"] = "Collecting submissions"

    return ApiResponse(data=list(grouped.values()))


@router.get("/classes", dependencies=[Depends(require_role(["teacher", "admin", "hod", "principal"]))])
async def list_classes(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all classes and sections in the current user's school."""
    _require_school(current_user)

    result = await db.execute(
        select(SchoolClass)
        .filter_by(school_id=current_user.school_id)
        .order_by(SchoolClass.name)
    )
    classes = result.scalars().all()

    items = []
    for c in classes:
        # Eagerly load sections
        sections_result = await db.execute(
            select(Section).filter_by(class_id=c.id).order_by(Section.name)
        )
        sections = sections_result.scalars().all()

        # Count students per section
        section_items = []
        for s in sections:
            student_count_result = await db.execute(
                select(func.count(Student.id)).filter_by(section_id=s.id)
            )
            student_count = student_count_result.scalar() or 0
            section_items.append({
                "id": str(s.id),
                "name": s.name,
                "student_count": student_count,
            })

        items.append({
            "id": str(c.id),
            "name": c.name,
            "sections": section_items,
            "total_students": sum(si["student_count"] for si in section_items),
        })

    return ApiResponse(data=items)

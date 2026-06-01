"""
Auth Service — JWT token management, password hashing, and user authentication.

Provides:
  - bcrypt password hashing/verification
  - JWT access/refresh token creation and validation
  - FastAPI dependency for extracting the current authenticated user
"""
import logging
from datetime import datetime, timedelta, timezone
import uuid
import httpx

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt, jwk
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import settings
from app.db.session import get_db
from app.db.models import User

logger = logging.getLogger(__name__)

# ── Password Hashing ──
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ── JWT Bearer Scheme ──
security = HTTPBearer()


def hash_password(password: str) -> str:
    """Hash a plain-text password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against its bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: str, role: str) -> str:
    """Create a JWT access token."""
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {
        "sub": user_id,
        "role": role,
        "type": "access",
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    """Create a JWT refresh token (longer-lived)."""
    expire = datetime.now(timezone.utc) + timedelta(
        days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
    )
    payload = {
        "sub": user_id,
        "type": "refresh",
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token. Raises HTTPException on failure."""
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Global in-memory cache for JWKS keys
_jwks_cache = None


async def fetch_jwks(supabase_url: str) -> dict:
    """Fetch and cache the JWKS keys from Supabase."""
    global _jwks_cache
    if _jwks_cache is not None:
        return _jwks_cache

    if not supabase_url:
        logger.warning("SUPABASE_URL is not configured. JWKS fetch skipped.")
        return None

    jwks_url = f"{supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(jwks_url, timeout=5.0)
            response.raise_for_status()
            _jwks_cache = response.json()
            logger.info("Successfully fetched and cached Supabase JWKS keys.")
            return _jwks_cache
    except Exception as e:
        logger.error(f"Failed to fetch JWKS keys from Supabase: {e}")
        return None


def decode_supabase_token(token: str, jwks: dict = None) -> dict:
    """
    Decodes and validates a Supabase JWT token using JWKS (public key) or fallback secret.
    """
    try:
        # Get unverified header to check the 'kid' and 'alg'
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        alg = unverified_header.get("alg", "HS256")
        
        # If JWKS is available and kid exists, attempt public key verification
        if jwks and kid:
            key_data = next((k for k in jwks.get("keys", []) if k.get("kid") == kid), None)
            if key_data:
                try:
                    key = jwk.construct(key_data)
                    return jwt.decode(
                        token,
                        key,
                        algorithms=[alg, "RS256", "ES256", "HS256"],
                        audience="authenticated"
                    )
                except Exception as e:
                    logger.debug(f"JWKS public key decoding failed, checking fallback secret: {e}")
                    if not settings.SUPABASE_JWT_SECRET:
                        raise e
        
        # Fallback to symmetric key verification if secret is configured
        if settings.SUPABASE_JWT_SECRET:
            return jwt.decode(
                token,
                settings.SUPABASE_JWT_SECRET,
                algorithms=[alg, "HS256", "RS256"],
                audience="authenticated"
            )
        
        # If neither worked, raise JWTError
        raise JWTError("No valid keys found for JWT verification.")
    except Exception as e:
        logger.error(f"JWT verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    FastAPI dependency — extracts and validates the Supabase JWT from the Authorization header,
    syncs/creates the User profile in the local database if needed, and returns the User.
    """
    token = credentials.credentials
    
    # 1. Decode token using Supabase JWKS or symmetric key secret
    jwks = None
    if settings.SUPABASE_URL:
        jwks = await fetch_jwks(settings.SUPABASE_URL)
    
    try:
        payload = decode_supabase_token(token, jwks)
    except HTTPException:
        # Graceful fallback to legacy local tokens if Supabase is not configured or for local test suites
        if not settings.SUPABASE_URL and not settings.SUPABASE_JWT_SECRET:
            payload = decode_token(token)
        else:
            raise

    user_id = payload.get("sub")
    email = payload.get("email")

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing user identifier.",
        )

    # Convert to UUID (Supabase uses UUIDs)
    try:
        user_uuid = uuid.UUID(user_id) if isinstance(user_id, str) else user_id
    except ValueError:
        # Fallback to normal string or mock UUID for legacy tokens if they are not standard UUIDs
        try:
            user_uuid = uuid.UUID(int=int(user_id))
        except Exception:
            user_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, str(user_id))

    # 2. Fetch user from local database
    result = await db.execute(select(User).filter_by(id=user_uuid))
    user = result.scalar_one_or_none()

    # 3. Auto-sync user if they exist in Supabase but not in the local Edexia db
    if not user:
        logger.info(f"Syncing new Supabase authenticated user: {email} ({user_uuid})")
        user_metadata = payload.get("user_metadata") or payload.get("raw_user_meta_data") or {}
        full_name = user_metadata.get("full_name") or user_metadata.get("name") or "Supabase User"
        role = user_metadata.get("role") or "teacher"

        # Concurrency safety: handle parallel API requests syncing the same user simultaneously
        try:
            user = User(
                id=user_uuid,
                email=email.lower() if email else f"{user_uuid}@supabase.temp",
                hashed_password="",  # managed by Supabase Auth
                full_name=full_name,
                role=role.lower(),
                is_active=True,
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
        except Exception as e:
            await db.rollback()
            # Race condition: another concurrent request inserted it first. Re-fetch.
            result = await db.execute(select(User).filter_by(id=user_uuid))
            user = result.scalar_one_or_none()
            if not user:
                logger.error(f"Failed to sync user {user_uuid} even after rollback/retry: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database error syncing user profile.",
                )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated.",
        )

    return user


def require_role(allowed_roles: list[str]):
    """
    FastAPI dependency factory for Role-Based Access Control (RBAC).
    Usage:
        @router.post("/", dependencies=[Depends(require_role(["admin", "principal"]))])
    """
    def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            logger.warning(f"RBAC Denied: User {current_user.id} ({current_user.role}) attempted to access restricted endpoint requiring {allowed_roles}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation not permitted. Required role: {', '.join(allowed_roles)}.",
            )
        return current_user
    
    return role_checker


# ── BOLA / IDOR Tenant Isolation Check Helpers ──
from sqlalchemy.orm import selectinload, joinedload
from app.db.models import Task, Submission, GradingRun, Student, ExamCycle, SchoolClass, Section, ClassroomWorksheet

async def check_task_access(task_id: uuid.UUID, user: User, db: AsyncSession) -> Task:
    """Check if task belongs to the user's school and return the task."""
    if user.school_id is None:
        result = await db.execute(select(Task).filter_by(id=task_id))
        task = result.scalar_one_or_none()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task

    result = await db.execute(
        select(Task)
        .options(joinedload(Task.exam_cycle))
        .filter_by(id=task_id)
    )
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Safe isolation check: only deny if both have non-null, different school IDs
    if task.exam_cycle and task.exam_cycle.school_id and task.exam_cycle.school_id != user.school_id:
        raise HTTPException(status_code=403, detail="Access denied: Task belongs to another school")
    
    return task

async def check_submission_access(submission_id: uuid.UUID, user: User, db: AsyncSession) -> Submission:
    """Check if submission belongs to the user's school and return it."""
    result = await db.execute(
        select(Submission)
        .options(
            joinedload(Submission.task).joinedload(Task.exam_cycle),
            joinedload(Submission.student).joinedload(Student.section).joinedload(Section.school_class)
        )
        .filter_by(id=submission_id)
    )
    sub = result.scalar_one_or_none()
    if not sub:
        # Fallback: check if this is a classroom worksheet
        ws_result = await db.execute(
            select(ClassroomWorksheet)
            .options(joinedload(ClassroomWorksheet.student).joinedload(Student.section).joinedload(Section.school_class))
            .filter_by(id=submission_id)
        )
        ws = ws_result.scalar_one_or_none()
        if ws:
            # Student Role Isolation: Enforce student only views their own worksheet
            if user.role == "student":
                from sqlalchemy import func
                email_name = user.email.split("@")[0].replace(".", " ").title()
                student_stmt = select(Student).filter(
                    (Student.id == user.id) |
                    (func.lower(Student.name) == func.lower(user.full_name)) |
                    (func.lower(Student.name) == func.lower(email_name))
                )
                student_res = await db.execute(student_stmt)
                students = student_res.scalars().all()
                student_ids = {s.id for s in students}
                student_ids.add(user.id)
                
                if ws.student_id not in student_ids:
                    raise HTTPException(status_code=403, detail="Access denied: Students can only access their own worksheets")
                    
            if user.school_id is not None:
                if ws.student and ws.student.section and ws.student.section.school_class and ws.student.section.school_class.school_id and ws.student.section.school_class.school_id != user.school_id:
                    raise HTTPException(status_code=403, detail="Access denied: Worksheet student belongs to another school")
            return ws
            
        raise HTTPException(status_code=404, detail="Submission not found")
        
    # Student Role Isolation: Enforce student only views their own submission
    if user.role == "student":
        from sqlalchemy import func
        email_name = user.email.split("@")[0].replace(".", " ").title()
        student_stmt = select(Student).filter(
            (Student.id == user.id) |
            (func.lower(Student.name) == func.lower(user.full_name)) |
            (func.lower(Student.name) == func.lower(email_name))
        )
        student_res = await db.execute(student_stmt)
        students = student_res.scalars().all()
        student_ids = {s.id for s in students}
        student_ids.add(user.id)
        
        if sub.student_id not in student_ids:
            raise HTTPException(status_code=403, detail="Access denied: Students can only access their own submissions")
            
    if user.school_id is not None:
        # Verify through task's exam cycle if exists and has a school_id
        if sub.task and sub.task.exam_cycle and sub.task.exam_cycle.school_id and sub.task.exam_cycle.school_id != user.school_id:
            raise HTTPException(status_code=403, detail="Access denied: Submission belongs to another school")
            
        # Or verify through student class if exists and has a school_id
        if sub.student and sub.student.section and sub.student.section.school_class and sub.student.section.school_class.school_id and sub.student.section.school_class.school_id != user.school_id:
            raise HTTPException(status_code=403, detail="Access denied: Student belongs to another school")

    return sub

async def check_run_access(run_id: uuid.UUID, user: User, db: AsyncSession) -> GradingRun:
    """Check if grading run belongs to the user's school and return it."""
    if user.school_id is None:
        result = await db.execute(select(GradingRun).filter_by(id=run_id))
        run = result.scalar_one_or_none()
        if not run:
            raise HTTPException(status_code=404, detail="Grading run not found")
        return run

    result = await db.execute(
        select(GradingRun)
        .options(joinedload(GradingRun.task).joinedload(Task.exam_cycle))
        .filter_by(id=run_id)
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Grading run not found")

    if run.task and run.task.exam_cycle and run.task.exam_cycle.school_id and run.task.exam_cycle.school_id != user.school_id:
        raise HTTPException(status_code=403, detail="Access denied: Grading run belongs to another school")

    return run

async def check_student_access(student_id: uuid.UUID, user: User, db: AsyncSession) -> Student:
    """Check if student belongs to the user's school and return it."""
    result = await db.execute(
        select(Student)
        .options(joinedload(Student.section).joinedload(Section.school_class))
        .filter_by(id=student_id)
    )
    student = result.scalar_one_or_none()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    if user.role == "student":
        from sqlalchemy import func
        email_name = user.email.split("@")[0].replace(".", " ").title()
        student_stmt = select(Student).filter(
            (Student.id == user.id) |
            (func.lower(Student.name) == func.lower(user.full_name)) |
            (func.lower(Student.name) == func.lower(email_name))
        )
        student_res = await db.execute(student_stmt)
        students = student_res.scalars().all()
        student_ids = {s.id for s in students}
        student_ids.add(user.id)
        
        if student_id not in student_ids:
            raise HTTPException(status_code=403, detail="Access denied: Students can only access their own student profile")

    if user.school_id is not None:
        if student.section and student.section.school_class and student.section.school_class.school_id and student.section.school_class.school_id != user.school_id:
            raise HTTPException(status_code=403, detail="Access denied: Student belongs to another school")

    return student


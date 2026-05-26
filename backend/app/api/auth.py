"""
Auth API — Signup, Login, Token Refresh, Profile, Preferences.

Public endpoints:
  - POST /auth/signup   — create account
  - POST /auth/login    — get JWT tokens
  - POST /auth/refresh  — refresh access token

Protected endpoints:
  - GET   /auth/me                 — get current user profile
  - PATCH /auth/me/preferences     — update confidence threshold & display name
  - PUT   /auth/gemini-key         — set/update BYOK Gemini API key
  - DELETE /auth/gemini-key        — remove stored Gemini key
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import User
from app.schemas.common import ApiResponse
from app.schemas.auth import (
    SignupRequest, LoginRequest, TokenResponse, RefreshRequest,
    UserResponse, UpdatePreferencesRequest,
)
from app.services.auth_service import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_token,
    get_current_user,
)
from app.config import settings
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request

logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ═══════════════════════════════════════════════════════════
# PUBLIC ENDPOINTS
# ═══════════════════════════════════════════════════════════

@router.post("/signup")
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def signup(request: Request, payload: SignupRequest, db: AsyncSession = Depends(get_db)):
    """Create a new user account."""
    # Check if email already exists
    result = await db.execute(select(User).filter_by(email=payload.email.lower()))
    existing = result.scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    # Validate role
    valid_roles = {"teacher", "admin", "student"}
    role = payload.role.lower()
    if role not in valid_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role '{payload.role}'. Must be one of: {', '.join(valid_roles)}",
        )

    # Create user
    user = User(
        email=payload.email.lower(),
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        role=role,
        is_active=True,
    )
    db.add(user)
    await db.flush()

    # Generate tokens
    access_token = create_access_token(str(user.id), user.role)
    refresh_token = create_refresh_token(str(user.id))

    logger.info(f"New user registered: {user.email} (role={user.role})")

    return ApiResponse(data={
        "user": UserResponse(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            has_gemini_key=False,
            is_active=True,
            created_at=user.created_at.isoformat(),
        ),
        "tokens": TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        ),
    })


@router.post("/login")
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def login(request: Request, payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate and get JWT tokens."""
    result = await db.execute(select(User).filter_by(email=payload.email.lower()))
    user = result.scalar_one_or_none()

    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Contact an administrator.",
        )

    access_token = create_access_token(str(user.id), user.role)
    refresh_token = create_refresh_token(str(user.id))

    logger.info(f"User logged in: {user.email}")

    return ApiResponse(data=TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    ))


@router.post("/refresh")
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def refresh_token(request: Request, payload: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """Refresh an access token using a valid refresh token."""
    token_data = decode_token(payload.refresh_token)

    if token_data.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid token type. Provide a refresh token.",
        )

    user_id = token_data.get("sub")
    result = await db.execute(select(User).filter_by(id=user_id))
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated.",
        )

    new_access = create_access_token(str(user.id), user.role)

    return ApiResponse(data=TokenResponse(
        access_token=new_access,
        refresh_token=payload.refresh_token,  # Reuse same refresh token
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    ))


# ═══════════════════════════════════════════════════════════
# PROTECTED ENDPOINTS
# ═══════════════════════════════════════════════════════════

@router.get("/me")
async def get_profile(current_user: User = Depends(get_current_user)):
    """Get the current user's profile."""
    return ApiResponse(data=UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        has_gemini_key=False,
        is_active=current_user.is_active,
        created_at=current_user.created_at.isoformat(),
        confidence_threshold=current_user.confidence_threshold or 0.75,
    ))


@router.patch("/me/preferences")
async def update_preferences(
    payload: UpdatePreferencesRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update per-user AI grading preferences.
    - confidence_threshold: 0.0–1.0. Grading runs below this percentage are auto-flagged for review.
    - full_name: Optional display name update.
    """
    current_user.confidence_threshold = payload.confidence_threshold
    if payload.full_name is not None:
        current_user.full_name = payload.full_name

    await db.commit()
    await db.refresh(current_user)

    logger.info(
        f"User {current_user.email} updated preferences: "
        f"confidence_threshold={payload.confidence_threshold}"
    )

    return ApiResponse(data=UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        has_gemini_key=False,
        is_active=current_user.is_active,
        created_at=current_user.created_at.isoformat(),
        confidence_threshold=current_user.confidence_threshold,
    ))

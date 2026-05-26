"""
Auth schemas — request/response models for authentication and user management.
"""
from pydantic import BaseModel, Field, EmailStr
from typing import Optional


class SignupRequest(BaseModel):
    """User registration schema."""
    email: str = Field(min_length=5, max_length=255, description="User email address")
    password: str = Field(min_length=8, max_length=128, description="Password (min 8 chars)")
    full_name: str = Field(min_length=1, max_length=255, description="Full name")
    role: str = Field(default="teacher", description="teacher | admin | student")


class LoginRequest(BaseModel):
    """User login schema."""
    email: str = Field(min_length=5, max_length=255)
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    """JWT token pair response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class RefreshRequest(BaseModel):
    """Token refresh schema."""
    refresh_token: str


class UserResponse(BaseModel):
    """User profile response."""
    id: str
    email: str
    full_name: str
    role: str
    has_gemini_key: bool = False
    is_active: bool
    created_at: str
    confidence_threshold: Optional[float] = 0.75

    model_config = {"from_attributes": True}


class UpdatePreferencesRequest(BaseModel):
    """Schema for updating per-user AI grading preferences."""
    confidence_threshold: float = Field(
        ge=0.0,
        le=1.0,
        description="Minimum AI confidence (0.0–1.0). Grading runs below this are flagged for review."
    )
    full_name: Optional[str] = Field(None, min_length=1, max_length=255)

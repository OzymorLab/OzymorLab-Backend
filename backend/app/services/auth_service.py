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
        # Get unverified header to check the 'kid'
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        
        # If JWKS is available and kid exists, attempt public key verification
        if jwks and kid:
            key_data = next((k for k in jwks.get("keys", []) if k.get("kid") == kid), None)
            if key_data:
                try:
                    key = jwk.construct(key_data)
                    return jwt.decode(
                        token,
                        key,
                        algorithms=["RS256", "ES256"],
                        audience="authenticated"
                    )
                except Exception as e:
                    logger.debug(f"JWKS public key decoding failed, checking fallback secret: {e}")
        
        # Fallback to symmetric key verification if secret is configured
        if settings.SUPABASE_JWT_SECRET:
            return jwt.decode(
                token,
                settings.SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
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

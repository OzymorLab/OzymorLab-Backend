"""
Ingestion service — handles file upload and storage via Supabase Storage.
"""
import uuid
import logging
import requests

from app.config import settings

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


def _get_auth_headers(content_type: str = None, user_token: str = None) -> dict:
    """Helper to build safe and backward-compatible Supabase REST headers."""
    # Prioritize the Service Role Key if configured in the backend (bypasses RLS perfectly!)
    if settings.SUPABASE_SERVICE_ROLE_KEY:
        headers = {
            "apikey": settings.SUPABASE_SERVICE_ROLE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
        }
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    headers = {
        "apikey": settings.SUPABASE_ANON_KEY,
    }
    if content_type:
        headers["Content-Type"] = content_type
        
    # If the user's JWT token is provided, prioritize it for Authorization!
    if user_token:
        if user_token.startswith("Bearer "):
            headers["Authorization"] = user_token
        else:
            headers["Authorization"] = f"Bearer {user_token}"
    # Otherwise, fallback to the anon key ONLY if it is not a new sb_publishable_ key (which is not a JWT)
    elif settings.SUPABASE_ANON_KEY and not settings.SUPABASE_ANON_KEY.startswith("sb_publishable_"):
        headers["Authorization"] = f"Bearer {settings.SUPABASE_ANON_KEY}"
        
    return headers


def ensure_bucket_exists(client=None) -> None:
    """
    Bucket validation helper.
    For Supabase Storage, we expect the bucket to be pre-created in the Supabase Dashboard.
    """
    logger.info(f"Using Supabase Storage Bucket: '{settings.SUPABASE_STORAGE_BUCKET}'")


def validate_file(filename: str, file_size: int) -> tuple[bool, str | None]:
    """Validate file type and size. Returns (is_valid, error_message)."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type: '.{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    if file_size > MAX_FILE_SIZE:
        return False, f"File too large: {file_size / (1024*1024):.1f}MB. Max: 20MB"
    return True, None


def upload_file(file_data: bytes, filename: str, content_type: str, folder: str = "submissions", user_token: str = None) -> str:
    """
    Upload a file to Supabase Storage and return the object key.

    Args:
        file_data: Raw file bytes
        filename: Original filename
        content_type: MIME type
        folder: Storage folder/prefix (default "submissions")
        user_token: Optional authenticated user's session JWT

    Returns:
        object_key: Supabase object path/key for retrieval
    """
    # Generate unique object key
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
    object_key = f"{folder}/{uuid.uuid4()}.{ext}"

    url = f"{settings.SUPABASE_URL}/storage/v1/object/{settings.SUPABASE_STORAGE_BUCKET}/{object_key}"
    headers = _get_auth_headers(content_type, user_token)

    # Upload using HTTP POST (or PUT fallback if there are conflicts)
    response = requests.post(url, data=file_data, headers=headers)
    if response.status_code not in (200, 201):
        # Fallback to PUT
        response = requests.put(url, data=file_data, headers=headers)

    if response.status_code not in (200, 201):
        logger.error(f"Supabase storage upload failed ({response.status_code}): {response.text}")
        raise Exception(f"Failed to upload file to Supabase Storage: {response.text}")

    return object_key


def download_file(object_key: str, user_token: str = None) -> bytes:
    """Download a file from Supabase Storage by its object key."""
    url = f"{settings.SUPABASE_URL}/storage/v1/object/authenticated/{settings.SUPABASE_STORAGE_BUCKET}/{object_key}"
    headers = _get_auth_headers(user_token=user_token)
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        logger.error(f"Supabase storage download failed ({response.status_code}): {response.text}")
        raise Exception(f"Failed to download file from Supabase Storage: {response.text}")
        
    return response.content


def delete_file(object_key: str, user_token: str = None) -> None:
    """Delete a file from Supabase Storage."""
    url = f"{settings.SUPABASE_URL}/storage/v1/object/{settings.SUPABASE_STORAGE_BUCKET}/{object_key}"
    headers = _get_auth_headers(user_token=user_token)
    
    try:
        response = requests.delete(url, headers=headers)
        if response.status_code not in (200, 204):
            logger.warning(f"Failed to delete file {object_key} from Supabase: {response.text}")
    except Exception as e:
        logger.error(f"Failed to delete file {object_key} from Supabase: {e}")


def generate_presigned_url(object_key: str, expiry_seconds: int = 3600, user_token: str = None) -> str:
    """
    Generate a signed URL for a Supabase Storage object.
    Used by the DEIS Diagram-marker to download submission images.

    Args:
        object_key: Supabase object path/key
        expiry_seconds: URL validity period (default 1 hour)
        user_token: Optional authenticated user's session JWT

    Returns:
        Fully qualified signed URL string
    """
    url = f"{settings.SUPABASE_URL}/storage/v1/object/sign/{settings.SUPABASE_STORAGE_BUCKET}/{object_key}"
    headers = _get_auth_headers("application/json", user_token)

    payload = {"expiresIn": expiry_seconds}

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            logger.error(f"Failed to create signed URL from Supabase ({response.status_code}): {response.text}")
            raise Exception(f"Failed to generate Supabase signed URL: {response.text}")

        data = response.json()
        signed_path = data.get("signedURL") or data.get("signedUrl")
        if not signed_path:
            raise Exception("Supabase signed URL response missing path")

        # If relative, construct the absolute URL
        if signed_path.startswith("/"):
            return f"{settings.SUPABASE_URL}/storage/v1{signed_path}"
        return signed_path
    except Exception as e:
        logger.error(f"Failed to generate signed URL for {object_key}: {e}")
        raise


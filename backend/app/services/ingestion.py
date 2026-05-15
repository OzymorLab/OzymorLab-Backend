"""
Ingestion service — handles file upload to MinIO and bucket management.
"""
import io
import uuid
from minio import Minio
from minio.error import S3Error

from app.config import settings


def get_minio_client() -> Minio:
    """Create a MinIO client instance."""
    return Minio(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_USER,
        secret_key=settings.MINIO_PASSWORD,
        secure=settings.MINIO_SECURE,
    )


def ensure_bucket_exists(client: Minio | None = None) -> None:
    """Create the submissions bucket if it doesn't exist."""
    client = client or get_minio_client()
    bucket = settings.MINIO_BUCKET
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


def validate_file(filename: str, file_size: int) -> tuple[bool, str | None]:
    """Validate file type and size. Returns (is_valid, error_message)."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type: '.{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    if file_size > MAX_FILE_SIZE:
        return False, f"File too large: {file_size / (1024*1024):.1f}MB. Max: 20MB"
    return True, None


def upload_file(file_data: bytes, filename: str, content_type: str) -> str:
    """
    Upload a file to MinIO and return the object key.

    Args:
        file_data: Raw file bytes
        filename: Original filename
        content_type: MIME type

    Returns:
        object_key: MinIO object key for retrieval
    """
    client = get_minio_client()
    ensure_bucket_exists(client)

    # Generate unique object key
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
    object_key = f"submissions/{uuid.uuid4()}.{ext}"

    # Upload to MinIO
    client.put_object(
        bucket_name=settings.MINIO_BUCKET,
        object_name=object_key,
        data=io.BytesIO(file_data),
        length=len(file_data),
        content_type=content_type,
    )

    return object_key


def download_file(object_key: str) -> bytes:
    """Download a file from MinIO by its object key."""
    client = get_minio_client()
    response = client.get_object(settings.MINIO_BUCKET, object_key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def delete_file(object_key: str) -> None:
    """Delete a file from MinIO."""
    client = get_minio_client()
    try:
        client.remove_object(settings.MINIO_BUCKET, object_key)
    except S3Error:
        pass  # File may not exist, that's fine

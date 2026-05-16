"""
Ingestion service — handles file upload to AWS S3.
"""
import uuid
import logging
import boto3
from botocore.exceptions import ClientError

from app.config import settings

logger = logging.getLogger(__name__)


def get_s3_client():
    """Create an AWS S3 client instance."""
    return boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION,
    )


def ensure_bucket_exists(client=None) -> None:
    """Create the S3 bucket if it doesn't exist."""
    client = client or get_s3_client()
    bucket = settings.S3_BUCKET
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            logger.info(f"Bucket {bucket} does not exist. Creating it...")
            if settings.AWS_REGION == "us-east-1":
                client.create_bucket(Bucket=bucket)
            else:
                client.create_bucket(
                    Bucket=bucket,
                    CreateBucketConfiguration={"LocationConstraint": settings.AWS_REGION},
                )
        else:
            logger.error(f"Error checking bucket {bucket}: {e}")
            raise


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
    Upload a file to AWS S3 and return the object key.

    Args:
        file_data: Raw file bytes
        filename: Original filename
        content_type: MIME type

    Returns:
        object_key: S3 object key for retrieval
    """
    client = get_s3_client()
    ensure_bucket_exists(client)

    # Generate unique object key
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
    object_key = f"submissions/{uuid.uuid4()}.{ext}"

    # Upload to S3
    client.put_object(
        Bucket=settings.S3_BUCKET,
        Key=object_key,
        Body=file_data,
        ContentType=content_type,
    )

    return object_key


def download_file(object_key: str) -> bytes:
    """Download a file from AWS S3 by its object key."""
    client = get_s3_client()
    response = client.get_object(Bucket=settings.S3_BUCKET, Key=object_key)
    return response["Body"].read()


def delete_file(object_key: str) -> None:
    """Delete a file from AWS S3."""
    client = get_s3_client()
    try:
        client.delete_object(Bucket=settings.S3_BUCKET, Key=object_key)
    except ClientError as e:
        logger.error(f"Failed to delete file {object_key}: {e}")

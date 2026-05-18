import pytest
from unittest.mock import patch, MagicMock
from app.services.ingestion import upload_file, download_file, delete_file, generate_presigned_url, validate_file
from app.config import settings

def test_validate_file():
    # Valid file
    is_valid, err = validate_file("test.pdf", 5 * 1024 * 1024)
    assert is_valid is True
    assert err is None

    # Invalid extension
    is_valid, err = validate_file("test.exe", 1024)
    assert is_valid is False
    assert "Invalid file type" in err

    # Too large
    is_valid, err = validate_file("test.pdf", 25 * 1024 * 1024)
    assert is_valid is False
    assert "File too large" in err

@patch("app.services.ingestion.requests.post")
def test_upload_file_success(mock_post):
    settings.SUPABASE_URL = "https://mock-supabase.co"
    settings.SUPABASE_ANON_KEY = "mock-key"
    settings.SUPABASE_STORAGE_BUCKET = "test-bucket"

    # Setup mock response
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"Key": "test-key"}
    mock_post.return_value = mock_response

    file_data = b"hello world"
    object_key = upload_file(file_data, "student_paper.pdf", "application/pdf")

    assert object_key.startswith("submissions/")
    assert object_key.endswith(".pdf")
    assert mock_post.called

    # Verify post arguments
    args, kwargs = mock_post.call_args
    assert args[0] == f"https://mock-supabase.co/storage/v1/object/test-bucket/{object_key}"
    assert kwargs["data"] == file_data
    assert kwargs["headers"]["Authorization"] == "Bearer mock-key"
    assert kwargs["headers"]["Content-Type"] == "application/pdf"

@patch("app.services.ingestion.requests.get")
def test_download_file_success(mock_get):
    settings.SUPABASE_URL = "https://mock-supabase.co"
    settings.SUPABASE_ANON_KEY = "mock-key"
    settings.SUPABASE_STORAGE_BUCKET = "test-bucket"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"downloaded bytes"
    mock_get.return_value = mock_response

    content = download_file("submissions/test-file.pdf")
    assert content == b"downloaded bytes"
    assert mock_get.called

    args, kwargs = mock_get.call_args
    assert args[0] == "https://mock-supabase.co/storage/v1/object/authenticated/test-bucket/submissions/test-file.pdf"
    assert kwargs["headers"]["Authorization"] == "Bearer mock-key"

@patch("app.services.ingestion.requests.delete")
def test_delete_file_success(mock_delete):
    settings.SUPABASE_URL = "https://mock-supabase.co"
    settings.SUPABASE_ANON_KEY = "mock-key"
    settings.SUPABASE_STORAGE_BUCKET = "test-bucket"

    mock_response = MagicMock()
    mock_response.status_code = 204
    mock_delete.return_value = mock_response

    delete_file("submissions/test-file.pdf")
    assert mock_delete.called

    args, kwargs = mock_delete.call_args
    assert args[0] == "https://mock-supabase.co/storage/v1/object/test-bucket/submissions/test-file.pdf"
    assert kwargs["headers"]["Authorization"] == "Bearer mock-key"

@patch("app.services.ingestion.requests.post")
def test_generate_presigned_url_success(mock_post):
    settings.SUPABASE_URL = "https://mock-supabase.co"
    settings.SUPABASE_ANON_KEY = "mock-key"
    settings.SUPABASE_STORAGE_BUCKET = "test-bucket"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"signedURL": "/storage/v1/object/sign/test-bucket/submissions/test-file.pdf?token=abc"}
    mock_post.return_value = mock_response

    url = generate_presigned_url("submissions/test-file.pdf", expiry_seconds=1800)
    assert url == "https://mock-supabase.co/storage/v1/object/sign/test-bucket/submissions/test-file.pdf?token=abc"
    assert mock_post.called

    args, kwargs = mock_post.call_args
    assert args[0] == "https://mock-supabase.co/storage/v1/object/sign/test-bucket/submissions/test-file.pdf"
    assert kwargs["json"] == {"expiresIn": 1800}
    assert kwargs["headers"]["Authorization"] == "Bearer mock-key"

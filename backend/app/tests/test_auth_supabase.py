import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException
from app.services.auth_service import decode_supabase_token, fetch_jwks, get_current_user
from app.config import settings
from jose import jwt

# Sample symmetric key for testing
TEST_SECRET = "test-jwt-secret-key-for-supabase-development-only"

@pytest.fixture
def mock_jwks():
    return {
        "keys": [
            {
                "kty": "oct",
                "kid": "test-kid",
                "use": "sig",
                "alg": "HS256",
                "k": "dGVzdC1qd3Qtc2VjcmV0LWtleS1mb3Itc3VwYWJhc2UtZGV2ZWxvcG1lbnQtb25seQ=="
            }
        ]
    }

def test_decode_supabase_token_symmetric():
    """Test symmetric decoding of Supabase tokens with a fallback secret."""
    settings.SUPABASE_JWT_SECRET = TEST_SECRET
    settings.SUPABASE_URL = ""
    
    # Generate mock token
    claims = {"sub": "8cf55255-a0cd-46a2-998f-897d9c614532", "email": "test@edexia.ai", "aud": "authenticated"}
    token = jwt.encode(claims, TEST_SECRET, algorithm="HS256")
    
    payload = decode_supabase_token(token)
    assert payload["sub"] == claims["sub"]
    assert payload["email"] == claims["email"]

@pytest.mark.asyncio
async def test_get_current_user_new_sync():
    """Test that a new Supabase user is automatically synced/created in the database."""
    settings.SUPABASE_URL = "https://mock-supabase.co"
    settings.SUPABASE_JWT_SECRET = TEST_SECRET

    user_uuid = uuid.uuid4()
    claims = {
        "sub": str(user_uuid),
        "email": "newuser@edexia.ai",
        "aud": "authenticated",
        "user_metadata": {
            "full_name": "New Supabase Student",
            "role": "student"
        }
    }
    
    token = jwt.encode(claims, TEST_SECRET, algorithm="HS256")
    
    # Mock JWT header and decode to bypass public key JWKS construct
    with patch("app.services.auth_service.jwt.get_unverified_header", return_value={"kid": "test-kid"}), \
         patch("app.services.auth_service.decode_supabase_token", return_value=claims), \
         patch("app.services.auth_service.fetch_jwks", return_value={}):
         
        # Mock DB async session
        db_mock = AsyncMock()
        
        # Create a synchronous MagicMock for the query result
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        
        # db.execute returns result_mock when awaited
        db_mock.execute.return_value = result_mock
        
        credentials_mock = MagicMock()
        credentials_mock.credentials = token
        
        # Run dependency
        user = await get_current_user(credentials=credentials_mock, db=db_mock)
        
        assert user.id == user_uuid
        assert user.email == "newuser@edexia.ai"
        assert user.full_name == "New Supabase Student"
        assert user.role == "student"
        assert user.is_active is True
        
        # Verify db insert was called
        assert db_mock.add.called
        assert db_mock.commit.called

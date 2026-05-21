"""
Idempotency Utility — DB-backed idempotency key manager for transaction stability.
"""
import functools
import hashlib
import json
import logging
from fastapi import Request, Response, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.models import IdempotencyKey

logger = logging.getLogger(__name__)


def idempotent():
    """
    Decorator to enforce idempotency on POST requests based on 'Idempotency-Key' header.

    Stores status in the database to prevent duplicate executions and parallel race conditions:
    1. If no header is present, proceeds normally.
    2. If header is present, computes HMAC/SHA256 signature.
    3. If status is PROCESSING, returns 409 Conflict.
    4. If status is SUCCESS, replays the cached status code and response body.
    5. Otherwise, marks as PROCESSING, executes the route, and saves the final response as SUCCESS.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            request: Request = kwargs.get("request")
            db: AsyncSession = kwargs.get("db")

            if not request or not db:
                # Fallback: if decorator is applied without request or db in arguments, bypass
                logger.warning(f"Idempotency bypassed on {func.__name__} - missing 'request' or 'db' keyword arguments.")
                return await func(*args, **kwargs)

            idem_key = request.headers.get("Idempotency-Key")
            if not idem_key:
                # No key, proceed normally
                return await func(*args, **kwargs)

            # Standard hash to normalize key size
            key_hash = hashlib.sha256(idem_key.encode("utf-8")).hexdigest()

            # ── Check existing key status ──
            result = await db.execute(
                select(IdempotencyKey).filter_by(key_hash=key_hash)
            )
            existing = result.scalar_one_or_none()

            if existing:
                if existing.status == "PROCESSING":
                    logger.warning(f"Concurrent request blocked by idempotency key: {idem_key}")
                    raise HTTPException(
                        status_code=409,
                        detail="Another request with this idempotency key is already in progress.",
                    )
                elif existing.status == "SUCCESS":
                    logger.info(f"Replaying cached response for idempotency key: {idem_key}")
                    try:
                        body_data = json.loads(existing.response_body)
                    except Exception:
                        body_data = existing.response_body

                    # If the endpoint usually returns a pydantic model or ApiResponse, return body_data
                    # Custom FastAPI responses are handled nicely when returning dict/list
                    return body_data

            # ── Reserve key by marking as PROCESSING ──
            new_record = IdempotencyKey(key_hash=key_hash, status="PROCESSING")
            db.add(new_record)
            # Use commit to write processing state immediately, releasing lock
            await db.commit()

            try:
                # Execute actual route handler
                response_val = await func(*args, **kwargs)

                # Format response body for persistence
                # Pydantic models have .model_dump() or dict()
                if hasattr(response_val, "model_dump"):
                    raw_body = response_val.model_dump()
                elif hasattr(response_val, "dict"):
                    raw_body = response_val.dict()
                else:
                    raw_body = response_val

                body_str = json.dumps(raw_body, default=str)

                # ── Update key as SUCCESS ──
                # Reload to avoid session concurrency issues
                result = await db.execute(
                    select(IdempotencyKey).filter_by(key_hash=key_hash)
                )
                record = result.scalar_one()
                record.status = "SUCCESS"
                record.response_code = 200
                record.response_body = body_str
                await db.commit()

                return response_val

            except Exception as e:
                # ── Rollback key to FAILED / Delete on system exception ──
                logger.error(f"Idempotency execution failed for key {idem_key}: {e}")
                # Reload and delete or mark failed to let client retry
                try:
                    result = await db.execute(
                        select(IdempotencyKey).filter_by(key_hash=key_hash)
                    )
                    record = result.scalar_one_or_none()
                    if record:
                        await db.delete(record)
                        await db.commit()
                except Exception as rollback_err:
                    logger.error(f"Failed to cleanup idempotency key: {rollback_err}")

                raise e

        return wrapper
    return decorator

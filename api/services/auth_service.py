"""Hash de senha, emissão e validação de JWT."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt

from api.config import settings
from api.models.user import UserRole

ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )
    except ValueError:
        return False


def _encode(payload: dict[str, Any], expires: datetime) -> str:
    body = {**payload, "exp": expires}
    return jwt.encode(body, settings.jwt_secret, algorithm=ALGORITHM)


def create_access_token(
    user_id: uuid.UUID,
    email: str,
    firm_id: uuid.UUID,
    role: UserRole,
) -> str:
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=settings.access_token_expire_minutes)
    return _encode(
        {
            "sub": str(user_id),
            "email": email,
            "firm_id": str(firm_id),
            "role": role.value,
            "typ": "access",
        },
        exp,
    )


def create_refresh_token(
    user_id: uuid.UUID,
    email: str,
    firm_id: uuid.UUID,
    role: UserRole,
) -> str:
    now = datetime.now(timezone.utc)
    exp = now + timedelta(days=settings.refresh_token_expire_days)
    return _encode(
        {
            "sub": str(user_id),
            "email": email,
            "firm_id": str(firm_id),
            "role": role.value,
            "typ": "refresh",
        },
        exp,
    )


def decode_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])


def verify_access_token(token: str) -> dict[str, Any]:
    data = decode_token(token)
    if data.get("typ") != "access":
        raise jwt.InvalidTokenError("Token não é access")
    return data


def verify_refresh_token(token: str) -> dict[str, Any]:
    data = decode_token(token)
    if data.get("typ") != "refresh":
        raise jwt.InvalidTokenError("Token não é refresh")
    return data

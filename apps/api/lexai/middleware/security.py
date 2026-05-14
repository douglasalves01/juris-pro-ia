from __future__ import annotations

import time
from typing import TYPE_CHECKING

import jwt
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import JSONResponse

from lexai.config import get_settings

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Receive, Scope, Send


class LexaiSecurityMiddleware:
    """Validates Supabase JWT (Bearer) and applies Redis-backed fixed-window rate limits."""

    def __init__(self, app: ASGIApp, *, fastapi_ref) -> None:
        self.app = app
        self.fastapi_ref = fastapi_ref

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path") or ""
        if path == "/health" or path.startswith("/docs") or path in (
            "/openapi.json",
            "/redoc",
            "/favicon.ico",
        ):
            await self.app(scope, receive, send)
            return

        settings = get_settings()
        headers = Headers(scope=scope)

        if settings.lexai_auth_disabled:
            scope.setdefault("state", {})
            scope["state"]["lexai_user"] = {
                "sub": "00000000-0000-0000-0000-000000000001",
                "email": "dev@lexai.local",
            }
        else:
            auth = headers.get("authorization")
            if not auth or not auth.lower().startswith("bearer "):
                resp = JSONResponse({"detail": "Missing or invalid Authorization header."}, status_code=401)
                await resp(scope, receive, send)
                return

            token = auth.split(" ", 1)[1].strip()
            if not settings.supabase_jwt_secret.strip():
                resp = JSONResponse(
                    {"detail": "Server auth is not configured (SUPABASE_JWT_SECRET)."},
                    status_code=500,
                )
                await resp(scope, receive, send)
                return

            try:
                payload = jwt.decode(
                    token,
                    settings.supabase_jwt_secret,
                    algorithms=["HS256"],
                    audience=settings.supabase_jwt_audience,
                    options={"require": ["exp", "sub"]},
                )
            except jwt.PyJWTError as exc:
                resp = JSONResponse({"detail": f"Invalid token: {exc!s}"}, status_code=401)
                await resp(scope, receive, send)
                return

            sub = str(payload.get("sub") or "")
            if not sub:
                resp = JSONResponse({"detail": "Token missing subject (sub)."}, status_code=401)
                await resp(scope, receive, send)
                return

            scope.setdefault("state", {})
            scope["state"]["lexai_user"] = {
                "sub": sub,
                "email": payload.get("email"),
                "role": payload.get("role"),
            }

        user = scope["state"]["lexai_user"]
        sub = str(user.get("sub") or "")

        redis = getattr(self.fastapi_ref.state, "redis", None)
        if redis is None:
            resp = JSONResponse({"detail": "Redis is not ready."}, status_code=503)
            await resp(scope, receive, send)
            return

        window = max(1, int(settings.rate_limit_window_seconds))
        bucket = int(time.time()) // window
        key = f"lexai:rl:{sub}:{bucket}"
        try:
            count = int(await redis.incr(key))
            if count == 1:
                await redis.expire(key, window)
        except Exception as exc:  # noqa: BLE001
            resp = JSONResponse({"detail": f"Rate limit store error: {exc!s}"}, status_code=503)
            await resp(scope, receive, send)
            return

        if count > int(settings.rate_limit_max):
            resp = JSONResponse(
                {
                    "detail": "Rate limit exceeded.",
                    "retry_after_seconds": window,
                },
                status_code=429,
                headers={"Retry-After": str(window)},
            )
            await resp(scope, receive, send)
            return

        await self.app(scope, receive, send)


def get_request_user(request: Request) -> dict:
    user = getattr(request.state, "lexai_user", None)
    if not isinstance(user, dict) or not user.get("sub"):
        raise RuntimeError("lexai_user missing from request.state; middleware ordering bug.")
    return user

from __future__ import annotations

from contextlib import asynccontextmanager

import asyncpg
import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lexai.config import get_settings
from lexai.db import create_pool
from lexai.middleware.security import LexaiSecurityMiddleware
from lexai.routers import chat as chat_router
from lexai.routers import health as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.pool = await create_pool(settings)
    app.state.redis = redis.from_url(settings.redis_url, decode_responses=True)
    yield
    pool: asyncpg.Pool = app.state.pool
    await pool.close()
    r: redis.Redis = app.state.redis
    await r.aclose()


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(title=settings.app_name, lifespan=lifespan)

    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["x-vercel-ai-ui-message-stream", "x-lexai-session-id"],
    )
    application.add_middleware(LexaiSecurityMiddleware, fastapi_ref=application)

    application.include_router(health_router.router)
    application.include_router(chat_router.router)

    return application


app = create_app()

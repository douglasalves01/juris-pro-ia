"""Registro, login e refresh de JWT."""

from __future__ import annotations

import uuid

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models.user import Firm, User, UserRole
from api.schemas.auth import (
    LoginRequest,
    LoginResponse,
    RefreshRequest,
    RegisterRequest,
)
from api.services.auth_service import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
    verify_refresh_token,
)
from api.services.document_service import normalize_cnpj

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> LoginResponse:
    cnpj = normalize_cnpj(body.cnpj)
    firm = Firm(
        id=uuid.uuid4(),
        name=body.firm_name.strip(),
        cnpj=cnpj,
        region=body.region.strip(),
    )
    user = User(
        id=uuid.uuid4(),
        firm_id=firm.id,
        email=body.email.lower().strip(),
        password_hash=hash_password(body.password),
        full_name=body.full_name.strip(),
        role=UserRole.admin,
        is_active=True,
    )
    db.add(firm)
    db.add(user)
    try:
        await db.flush()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="E-mail ou CNPJ já cadastrado",
        )
    await db.refresh(user)
    return LoginResponse(
        access_token=create_access_token(
            user.id, user.email, user.firm_id, user.role
        ),
        refresh_token=create_refresh_token(
            user.id, user.email, user.firm_id, user.role
        ),
    )


@router.post("/login", response_model=LoginResponse)
async def login(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> LoginResponse:
    r = await db.execute(select(User).where(User.email == body.email.lower().strip()))
    user = r.scalar_one_or_none()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciais inválidas",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Usuário inativo",
        )
    return LoginResponse(
        access_token=create_access_token(
            user.id, user.email, user.firm_id, user.role
        ),
        refresh_token=create_refresh_token(
            user.id, user.email, user.firm_id, user.role
        ),
    )


@router.post("/refresh", response_model=LoginResponse)
async def refresh_tokens(
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> LoginResponse:
    try:
        data = verify_refresh_token(body.refresh_token)
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token inválido",
        )
    try:
        uid = uuid.UUID(data["sub"])
    except (KeyError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token inválido",
        )
    user = await db.get(User, uid)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuário inativo ou inexistente",
        )
    return LoginResponse(
        access_token=create_access_token(
            user.id, user.email, user.firm_id, user.role
        ),
        refresh_token=create_refresh_token(
            user.id, user.email, user.firm_id, user.role
        ),
    )

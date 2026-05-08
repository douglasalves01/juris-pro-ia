"""CRUD de usuários do escritório."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.dependencies import get_current_user, get_user_by_id_same_firm, require_roles
from api.models.user import User, UserRole
from api.schemas.user import UserCreate, UserResponse, UserUpdate
from api.services.auth_service import hash_password

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/", response_model=list[UserResponse])
async def list_users(
    db: AsyncSession = Depends(get_db),
    current: User = Depends(
        require_roles(UserRole.admin, UserRole.advogado, UserRole.secretaria)
    ),
) -> list[User]:
    r = await db.execute(select(User).where(User.firm_id == current.firm_id))
    return list(r.scalars().all())


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    body: UserCreate,
    db: AsyncSession = Depends(get_db),
    current: User = Depends(require_roles(UserRole.admin)),
) -> User:
    user = User(
        id=uuid.uuid4(),
        firm_id=current.firm_id,
        email=body.email.lower().strip(),
        password_hash=hash_password(body.password),
        full_name=body.full_name.strip(),
        role=body.role,
        is_active=True,
    )
    db.add(user)
    try:
        await db.flush()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="E-mail já cadastrado",
        )
    await db.refresh(user)
    return user


@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: uuid.UUID,
    body: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current: User = Depends(require_roles(UserRole.admin)),
) -> User:
    target = await get_user_by_id_same_firm(user_id, db, current)
    if body.full_name is not None:
        target.full_name = body.full_name.strip()
    if body.role is not None:
        target.role = body.role
    if body.is_active is not None:
        target.is_active = body.is_active
    await db.flush()
    await db.refresh(target)
    return target


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current: User = Depends(require_roles(UserRole.admin)),
) -> None:
    if user_id == current.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Não é possível excluir o próprio usuário",
        )
    target = await get_user_by_id_same_firm(user_id, db, current)
    db.delete(target)
    await db.flush()

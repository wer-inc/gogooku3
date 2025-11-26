"""User management endpoints."""

from app.models.user import User, UserCreate, UserUpdate
from app.services.user_service import UserService
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter()


def get_user_service() -> UserService:
    """Dependency injection for UserService."""
    return UserService()


@router.get("", response_model=list[User])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    service: UserService = Depends(get_user_service),
):
    """List all users with pagination."""
    return await service.get_users(skip=skip, limit=limit)


@router.get("/{user_id}", response_model=User)
async def get_user(
    user_id: int,
    service: UserService = Depends(get_user_service),
):
    """Get a specific user by ID."""
    user = await service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.post("", response_model=User)
async def create_user(
    user: UserCreate,
    service: UserService = Depends(get_user_service),
):
    """Create a new user."""
    return await service.create_user(user)


@router.put("/{user_id}", response_model=User)
async def update_user(
    user_id: int,
    user: UserUpdate,
    service: UserService = Depends(get_user_service),
):
    """Update an existing user."""
    updated = await service.update_user(user_id, user)
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    return updated


@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    service: UserService = Depends(get_user_service),
):
    """Delete a user."""
    deleted = await service.delete_user(user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted"}

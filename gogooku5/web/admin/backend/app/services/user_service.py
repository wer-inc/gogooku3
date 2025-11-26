"""User service for business logic."""

from datetime import datetime
from typing import Optional

from app.models.user import User, UserCreate, UserUpdate


class UserService:
    """User service with business logic."""

    # In-memory storage for demo (replace with DB in production)
    _users: dict[int, dict] = {
        1: {
            "id": 1,
            "email": "admin@example.com",
            "username": "admin",
            "is_active": True,
            "is_admin": True,
            "created_at": datetime.now(),
            "updated_at": None,
        }
    }
    _next_id: int = 2

    async def get_users(self, skip: int = 0, limit: int = 100) -> list[User]:
        """Get all users with pagination."""
        users = list(self._users.values())[skip : skip + limit]
        return [User(**u) for u in users]

    async def get_user(self, user_id: int) -> Optional[User]:
        """Get a user by ID."""
        user_data = self._users.get(user_id)
        if user_data:
            return User(**user_data)
        return None

    async def create_user(self, user: UserCreate) -> User:
        """Create a new user."""
        user_id = self._next_id
        self._next_id += 1

        user_data = {
            "id": user_id,
            "email": user.email,
            "username": user.username,
            "is_active": user.is_active,
            "is_admin": user.is_admin,
            "created_at": datetime.now(),
            "updated_at": None,
            # Note: In production, hash the password!
        }
        self._users[user_id] = user_data
        return User(**user_data)

    async def update_user(self, user_id: int, user: UserUpdate) -> Optional[User]:
        """Update an existing user."""
        if user_id not in self._users:
            return None

        user_data = self._users[user_id]
        update_data = user.model_dump(exclude_unset=True)

        for key, value in update_data.items():
            if key != "password" and value is not None:
                user_data[key] = value

        user_data["updated_at"] = datetime.now()
        return User(**user_data)

    async def delete_user(self, user_id: int) -> bool:
        """Delete a user."""
        if user_id in self._users:
            del self._users[user_id]
            return True
        return False

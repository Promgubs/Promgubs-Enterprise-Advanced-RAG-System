import uuid
from datetime import timedelta

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from .config import get_settings
from .db import User, get_db, utcnow

ADMIN_ROLE = "admin"
EDITOR_ROLE = "editor"  # may upload documents
PUBLIC_ROLE = "all"  # documents tagged "all" are visible to every authenticated user

oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_token(user: User) -> str:
    s = get_settings()
    exp = utcnow() + timedelta(minutes=s.access_token_minutes)
    return jwt.encode({"sub": str(user.id), "exp": exp}, s.jwt_secret, algorithm=s.jwt_algorithm)


def current_user(token: str = Depends(oauth2), db: Session = Depends(get_db)) -> User:
    s = get_settings()
    unauthorized = HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    try:
        payload = jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])
        user = db.get(User, uuid.UUID(payload["sub"]))
    except (jwt.PyJWTError, KeyError, ValueError):
        raise unauthorized
    if user is None:
        raise unauthorized
    # Roles are read from the database on every request, not from the token,
    # so removing a role takes effect immediately.
    return user


def is_admin(user) -> bool:
    return ADMIN_ROLE in user.roles


def effective_roles(user) -> list[str]:
    return sorted({r.lower() for r in user.roles} | {PUBLIC_ROLE})


def can_access(user, allowed_roles: list[str]) -> bool:
    return is_admin(user) or bool(set(effective_roles(user)) & set(allowed_roles))


def require_admin(user: User = Depends(current_user)) -> User:
    if not is_admin(user):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin only")
    return user


def require_editor(user: User = Depends(current_user)) -> User:
    if not (is_admin(user) or EDITOR_ROLE in user.roles):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Editor role required")
    return user

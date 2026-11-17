from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import User, get_db
from ..schemas import RolesUpdate, Token, UserCreate, UserOut
from ..security import create_token, current_user, hash_password, require_admin, verify_password

router = APIRouter(tags=["auth"])


def _roles(raw: list[str]) -> list[str]:
    return sorted({r.strip().lower() for r in raw if r.strip()})


@router.post("/auth/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == form.username.strip().lower()))
    if user is None or not verify_password(form.password, user.password_hash):
        raise HTTPException(401, "Incorrect email or password")
    return Token(access_token=create_token(user))


@router.get("/users/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return user


@router.post("/users", response_model=UserOut, status_code=201, dependencies=[Depends(require_admin)])
def create_user(body: UserCreate, db: Session = Depends(get_db)):
    email = body.email.strip().lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "Email already registered")
    user = User(email=email, password_hash=hash_password(body.password), roles=_roles(body.roles))
    db.add(user)
    db.commit()
    return user


@router.put("/users/{user_id}/roles", response_model=UserOut, dependencies=[Depends(require_admin)])
def set_roles(user_id: str, body: RolesUpdate, db: Session = Depends(get_db)):
    import uuid

    try:
        user = db.get(User, uuid.UUID(user_id))
    except ValueError:
        user = None
    if user is None:
        raise HTTPException(404, "User not found")
    user.roles = _roles(body.roles)
    db.commit()
    return user

# app/core/crud.py

from sqlalchemy.orm import Session
from app.models.users import User


def get_user_by_username(db: Session, username: str) -> User | None:
    """
    Fetches a user from the database by their username.
    """
    return db.query(User).filter(User.username == username).first()


# You can also move your 'create_user' logic from routes_auth.py
# into this file later to keep all your user CRUD logic in one place.

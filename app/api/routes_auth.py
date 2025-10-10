"""
routes_auth.py
---------------
Authentication routes for EduForge.

Features:
- Login endpoint (validates username & password).
- Issues JWT access tokens upon successful authentication.
- Uses Argon2 password hashing from security.py.
- Currently uses an in-memory "fake_users" dictionary
  (replace with real DB integration later).
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from app.core import security

# Create a router instance
router = APIRouter(prefix="/auth", tags=["Authentication"])

# ------------------------------------------------
# Fake user "database" (for testing/demo purposes)
# ------------------------------------------------
# TODO: Replace with a real database (e.g. PostgreSQL + SQLAlchemy models)
fake_users = {
    "admin": security.get_password_hash("admin"),         # password: "admin"
    "precious": security.get_password_hash("admin"),      # password: "admin"
    "teacher": security.get_password_hash("password123"), # password: "password123"
}

# ------------------------------------------------
# Request Models
# ------------------------------------------------
class LoginRequest(BaseModel):
    username: str
    password: str

# ------------------------------------------------
# Routes
# ------------------------------------------------
@router.post("/login")
def login(request: LoginRequest):
    """
    Authenticate user and return a JWT token.

    Args:
        request (LoginRequest): Username & password payload.

    Returns:
        dict: {"access_token": <JWT>, "token_type": "bearer"}

    Raises:
        HTTPException: If credentials are invalid.
    """
    # Fetch hashed password for this user (None if user does not exist)
    user_pw = fake_users.get(request.username)

    # Check if user exists AND password matches
    if not user_pw or not security.verify_password(request.password, user_pw):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Generate JWT token with username as "sub"
    token = security.create_access_token(data={"sub": request.username})

    return {"access_token": token, "token_type": "bearer"}


# from fastapi import APIRouter, HTTPException, Depends
# from pydantic import BaseModel
# from app.core import security

# router = APIRouter()

# # TODO: Replace with real DB
# fake_users = {
#     "admin": security.get_password_hash("admin"),
#     "precious": security.get_password_hash("admin"),
#     "teacher": security.get_password_hash("password123"),
# }



# class LoginRequest(BaseModel):
#     username: str
#     password: str


# @router.post("/login")
# def login(request: LoginRequest):
#     user_pw = fake_users.get(request.username)
#     if not user_pw or not security.verify_password(request.password, user_pw):
#         raise HTTPException(status_code=401, detail="Invalid credentials")

#     token = security.create_access_token(data={"sub": request.username})
#     return {"access_token": token, "token_type": "bearer"}

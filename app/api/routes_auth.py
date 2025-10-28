# app/routes/routes_auth.py

"""
routes_auth.py
---------------
Authentication routes for EduForge.

Handles user registration and login by connecting
to the real database.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from sqlalchemy import or_  # Import 'or_' for flexible queries

from app.core import security
from app.services.database import get_db
import app.schema.users as schemas  # Import your Pydantic schemas
import app.models.users as models  # Import your SQLAlchemy model

# --- 1. IMPORT THE NEW MODEL ---
from app.subscription_model import UsageLimits


# Create a router instance
router = APIRouter(prefix="/auth", tags=["Authentication"])

# ------------------------------------------------
# Routes
# ------------------------------------------------


@router.post("/register", response_model=schemas.User)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user in the database.
    """
    # Check if user with this username or email already exists
    db_user = (
        db.query(models.User)
        .filter(
            or_(models.User.username == user.username, models.User.email == user.email)
        )
        .first()
    )

    if db_user:
        if db_user.username == user.username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered",
            )
        if db_user.email == user.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

    # Hash the password
    hashed_password = security.get_password_hash(user.password)

    # --- 2. UPDATE USER CREATION LOGIC ---

    # Create new User model instance
    new_user = models.User(
        username=user.username, email=user.email, hashed_password=hashed_password
    )

    # Create their associated usage limits
    new_usage = UsageLimits()

    # Link them together via the relationship
    # This tells SQLAlchemy to create both
    new_user.usage = new_usage

    # Add to session, commit, and refresh
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    # -------------------------------------

    return new_user


@router.post("/token", response_model=schemas.Token)
def login_for_access_token(
    request: schemas.LoginRequest, db: Session = Depends(get_db)
):
    """
    Authenticate user and return a JWT token.
    Matches the '/auth/token' endpoint called by the frontend.

    Accepts either a username or an email as the 'username' field.
    """
    # Find user by EITHER username or email
    user = (
        db.query(models.User)
        .filter(
            or_(
                models.User.username == request.username,
                models.User.email == request.username,
            )
        )
        .first()
    )

    # Check if user exists AND password matches
    if not user or not security.verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",  # Use a generic message for security
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Generate JWT token with username as "sub"
    token = security.create_access_token(data={"sub": user.username})

    return {"access_token": token, "token_type": "bearer"}


# """
# routes_auth.py
# ---------------
# Authentication routes for EduForge.

# Handles user registration and login by connecting
# to the real database.
# """

# from fastapi import APIRouter, HTTPException, Depends, status
# from sqlalchemy.orm import Session
# from sqlalchemy import or_  # Import 'or_' for flexible queries

# from app.core import security
# from app.services.database import get_db
# import app.schema.users as schemas  # Import your Pydantic schemas
# import app.models.users as models  # Import your SQLAlchemy model

# # Create a router instance
# router = APIRouter(prefix="/auth", tags=["Authentication"])

# # ------------------------------------------------
# # Routes
# # ------------------------------------------------


# @router.post("/register", response_model=schemas.User)
# def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
#     """
#     Register a new user in the database.
#     """
#     # Check if user with this username or email already exists
#     db_user = (
#         db.query(models.User)
#         .filter(
#             or_(models.User.username == user.username, models.User.email == user.email)
#         )
#         .first()
#     )

#     if db_user:
#         if db_user.username == user.username:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Username already registered",
#             )
#         if db_user.email == user.email:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Email already registered",
#             )

#     # Hash the password
#     hashed_password = security.get_password_hash(user.password)

#     # Create new User model instance
#     new_user = models.User(
#         username=user.username, email=user.email, hashed_password=hashed_password
#     )

#     # Add to session, commit, and refresh
#     db.add(new_user)
#     db.commit()
#     db.refresh(new_user)

#     return new_user


# @router.post("/token", response_model=schemas.Token)
# def login_for_access_token(
#     request: schemas.LoginRequest, db: Session = Depends(get_db)
# ):
#     """
#     Authenticate user and return a JWT token.
#     Matches the '/auth/token' endpoint called by the frontend.

#     Accepts either a username or an email as the 'username' field.
#     """
#     # Find user by EITHER username or email
#     user = (
#         db.query(models.User)
#         .filter(
#             or_(
#                 models.User.username == request.username,
#                 models.User.email == request.username,
#             )
#         )
#         .first()
#     )

#     # Check if user exists AND password matches
#     if not user or not security.verify_password(request.password, user.hashed_password):
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid username or password",  # Use a generic message for security
#             headers={"WWW-Authenticate": "Bearer"},
#         )

#     # Generate JWT token with username as "sub"
#     token = security.create_access_token(data={"sub": user.username})

#     return {"access_token": token, "token_type": "bearer"}

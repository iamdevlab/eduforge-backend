"""
security.py
------------
Handles password hashing, JWT token creation, and authentication utilities
for the EduForge backend.

Notes:
- Using Argon2 instead of bcrypt for password hashing (modern, secure, no 72-char limit).
- Provides functions for hashing/verifying passwords.
- Provides functions for creating and validating JWT tokens.
"""

from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext

# Load constants from app configuration
from app.core.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

# -------------------------
# Password Hashing
# -------------------------
# Passlib context configured to use Argon2 (preferred modern algorithm).
pwd_context = CryptContext(
    schemes=["argon2"],   # use Argon2 instead of bcrypt
    deprecated="auto"     # auto-mark old algorithms as deprecated if added later
)

def get_password_hash(password: str) -> str:
    """
    Hashes a plain-text password using Argon2.

    Args:
        password (str): Plain text password.

    Returns:
        str: Secure Argon2 hash of the password.
    """
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    """
    Verifies a plain-text password against a hashed password.

    Args:
        plain (str): Plain text password entered by the user.
        hashed (str): Previously stored hashed password.

    Returns:
        bool: True if the password matches, False otherwise.
    """
    return pwd_context.verify(plain, hashed)

# -------------------------
# JWT Token Handling
# -------------------------
def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    """
    Creates a signed JWT access token.

    Args:
        data (dict): Data to encode into the token (e.g. {"sub": username}).
        expires_delta (timedelta, optional): Custom expiration time.
                                             Defaults to ACCESS_TOKEN_EXPIRE_MINUTES.

    Returns:
        str: Encoded JWT token as a string.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# -------------------------
# JWT Token Verification
# -------------------------
# OAuth2PasswordBearer tells FastAPI where to find the login endpoint
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """
    Decodes and verifies a JWT token, extracting the current user's username.

    Args:
        token (str): JWT token provided in the request Authorization header.

    Returns:
        str: The username (subject) extracted from the token.

    Raises:
        HTTPException: If the token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

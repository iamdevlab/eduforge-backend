# app/schemas/users.py

import datetime
from pydantic import BaseModel, EmailStr, ConfigDict
from typing import Optional  # <-- Added Optional
from app.subscription_model import SubscriptionTier  # <-- Added import


# --- NEW SCHEMA ---
# This schema will be nested in the User response
class UsageLimits(BaseModel):
    lesson_notes_generated: int
    exam_questions_generated: int

    class Config:
        model_config = ConfigDict(from_attributes=True)


# --- User Schemas ---


class UserBase(BaseModel):
    """
    Base schema for a User. Contains fields that are common
    for both creation and reading.
    """

    username: str
    email: EmailStr


class UserCreate(UserBase):
    """
    Schema used when creating a new user (at /register).
    It inherits username/email and adds the password.
    """

    password: str


class User(UserBase):
    """
    Schema used when reading/returning user data from the API.
    It includes the ID and creation time but omits the password.
    """

    id: int
    created_at: datetime.datetime

    # --- NEW FIELDS ADDED TO THE RESPONSE ---
    subscription_tier: SubscriptionTier
    subscription_expires_at: Optional[datetime.datetime]
    usage: Optional[UsageLimits]  # <-- Nested schema
    # ----------------------------------------

    class Config:
        model_config = ConfigDict(
            from_attributes=True
        )  # Tells Pydantic to read data from ORM models


# --- Auth Schemas ---


class LoginRequest(BaseModel):
    """
    Schema for the /login endpoint.
    'username' field will accept either a username or an email.
    """

    username: str  # This field is for "username_or_email"
    password: str


class Token(BaseModel):
    """
    Schema for returning the JWT token.
    """

    access_token: str
    token_type: str

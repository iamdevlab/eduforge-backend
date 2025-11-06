# app/subscription_model.py

import enum
from sqlalchemy import Column, Integer, ForeignKey, Enum
from sqlalchemy.orm import relationship
from app.services.database import Base  # Import Base from your database.py file


#
class SubscriptionTier(enum.Enum):
    """Enumeration for subscription tiers."""

    FREE = "free"
    PRO = "pro"


class UsageLimits(Base):
    """
    A new table to track usage for FREE-tier users.
    This is a one-to-one relationship with the User.
    """

    __tablename__ = "usage_limits"

    # This is both the Primary Key and a Foreign Key
    # We use "users.id" as a string to find the table.
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)

    lesson_notes_generated = Column(Integer, nullable=False, default=0)
    exam_questions_generated = Column(Integer, nullable=False, default=0)

    # --- RELATIONSHIP ---
    # This links back to the User model
    # We use "User" as a string to avoid import errors.
    owner = relationship("User", back_populates="usage")

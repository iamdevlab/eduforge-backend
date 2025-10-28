# app/core/dependencies.py

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

# Import your auth function and DB session
from app.core.security import get_current_user
from app.services.database import get_db

# Import your models
from app.models.users import User
from app.subscription_model import SubscriptionTier

# Import the helper we just created
from app.core.crud import get_user_by_username


# --- 1. New Dependency to get the full User object ---
# Your get_current_user() returns a username, but we need the
# full User database object to check their plan.
def get_current_db_user(
    username: str = Depends(get_current_user), db: Session = Depends(get_db)
) -> User:
    """
    Dependency to get the full User object from the database
    using the username from the JWT token.
    """
    user = get_user_by_username(db, username=username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    # This loads the 'usage' relationship at the same time
    # It's more efficient than letting SQLAlchemy auto-load it later
    if user.usage is None:
        # This is a fallback in case the registration failed
        # to create a usage row.
        pass

    return user


# --- 2. New Dependency to check Lesson Plan limits ---
def check_lesson_plan_limit(user: User = Depends(get_current_db_user)):
    """
    A dependency that checks if a user has hit their
    lesson plan generation limit.
    """
    # Pro users are always allowed
    if user.subscription_tier == SubscriptionTier.PRO:
        return

    # Free users are checked
    # The 'user.usage' relationship was loaded in get_current_db_user
    if user.usage and user.usage.lesson_notes_generated >= 2:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="You have reached your free limit of 2 lesson plan. Please upgrade to Pro.",
        )
    return


# --- 3. New Dependency to check Exam Question limits ---
def check_exam_question_limit(user: User = Depends(get_current_db_user)):
    """
    A dependency that checks if a user has hit their
    exam question generation limit.
    """
    # Pro users are always allowed
    if user.subscription_tier == SubscriptionTier.PRO:
        return

    # Free users are checked
    if user.usage and user.usage.exam_questions_generated >= 5:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="You have reached your free limit of 5 exam questions. Please upgrade your Plan.",
        )
    return

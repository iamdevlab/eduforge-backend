# app/api/lesson_plan.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Union
from datetime import date
import logging

# --- IMPORT MODELS AND THE GENERATOR FUNCTION ---
from app.services.ai_lesson_plan_generator import (
    generate_lesson_plan,
    _generate_week_entry,
    GenerationMetrics,
)
from app.models.lesson_plan_model import (
    LessonPlan,
    LessonPlanRequest,
    LessonWeek,
    WeekGenerationError,
)

from sqlalchemy.orm import Session
from app.models.users import User
from app.subscription_model import SubscriptionTier
from app.services.database import get_db
from app.core.dependencies import get_current_db_user, check_lesson_plan_limit
# ------------------------------

router = APIRouter()

# -------------------------
# Logging Configuration
# -------------------------
logger = logging.getLogger("lesson_plan_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# -------------------------
# Response Model
# -------------------------
class LessonPlanResponse(BaseModel):
    plan: LessonPlan

    class Config:
        from_attributes = True  #


# -------------------------
# Helper: summarize fallbacks and enrichments
# -------------------------
def summarize_failures(weeks) -> dict:  # Renamed for clarity
    failed_count = 0
    enrichment_count = 0

    for w in weeks:
        if w.status == "failed":  # <-- UPDATED LOGIC
            failed_count += 1
        # Check for success before accessing topic
        elif w.status == "success" and "Enrichment / consolidation" in (
            getattr(w, "topic", "") or ""
        ):
            enrichment_count += 1

    ai_generated_weeks = len(weeks) - failed_count - enrichment_count
    return {
        "total_weeks": len(weeks),
        "failed_weeks": failed_count,  # <-- Renamed
        "enrichment_weeks": enrichment_count,
        "ai_generated_weeks": ai_generated_weeks,
    }


# -------------------------
# Protected Endpoint
# -------------------------
@router.post(
    "/lesson-plan",
    response_model=LessonPlanResponse,
    summary="Generate AI-powered lesson plan",
    # --- 2. ADD THE DEPENDENCY ---
    dependencies=[Depends(check_lesson_plan_limit)],
)
async def create_lesson_plan(
    req: LessonPlanRequest,
    # --- 3. UPDATE THE FUNCTION SIGNATURE ---
    user: User = Depends(get_current_db_user),  # Get the full User object
    db: Session = Depends(get_db),  # Get the DB session
):
    """
    Generate a complete lesson plan for a subject, class, and term.
    """

    try:
        # Handle comma-separated topics from frontend
        if isinstance(req.topics, str):  #
            req.topics = [t.strip() for t in req.topics.split(",") if t.strip()]

        # Limit topics to duration weeks
        req.topics = req.topics[: req.duration_weeks or 10]
        duration_weeks = min(req.duration_weeks or 10, 12)

        # Generate the lesson plan via AI
        plan = await generate_lesson_plan(
            school_name=req.school_name,
            state=req.state,
            lga=req.lga,
            subject=req.subject,
            class_level=req.class_level,
            term=req.term,
            resumption_date=req.resumption_date,
            duration_weeks=duration_weeks,
            topics=req.topics,
        )

        # --- 4. CRITICAL: INCREMENT THE COUNT ---
        if user.subscription_tier == SubscriptionTier.FREE:
            # user.usage is the UsageLimits object
            user.usage.lesson_notes_generated += 1
            db.commit()
        # ----------------------------------------

        # Log generation details
        summary = summarize_failures(plan.weeks)
        logger.info(
            f"Lesson plan generated for {req.subject} ({req.class_level}) by {user.username}. Summary: {summary}"
        )

        return LessonPlanResponse(plan=plan)

    except Exception as e:
        logger.exception("Failed to generate lesson plan")
        raise HTTPException(
            status_code=500, detail=f"Lesson plan generation failed: {str(e)}"
        )


# -------------------------
# --- 5. NEW ENDPOINT FOR RETRYING A SINGLE WEEK ---
# -------------------------
class RetryWeekRequest(BaseModel):
    """The data needed to retry a single week's generation."""

    subject: str
    class_level: str
    term: str
    week_number: int
    start_date: date
    end_date: date
    topic: str


class DummyMetrics:
    """A dummy metrics object for single retries."""

    def record_success(self):
        logger.info("Retry week succeeded.")

    def record_failure(self):
        logger.warning("Retry week failed.")


@router.post(
    "/lesson-plan/retry-week",
    response_model=Union[LessonWeek, WeekGenerationError],
    summary="Retry a single failed lesson week",
)
async def retry_lesson_week(
    request: RetryWeekRequest,
    user: User = Depends(get_current_db_user),  # <-- SECURED: User must be logged in
):
    """
    Retries the generation for a single failed week.
    This endpoint does NOT count against the user's generation quota.
    """
    logger.info(
        f"User {user.username} retrying week {request.week_number} ('{request.topic}')"
    )

    try:
        # Prepare arguments for the generator function
        week_meta = {"start_date": request.start_date, "end_date": request.end_date}
        dummy_metrics = DummyMetrics()

        # Call the single-week generator directly.
        # The @retry_on_ai_error decorator is still active on it.
        result = await _generate_week_entry(
            week_idx=request.week_number,
            week_meta=week_meta,
            topic=request.topic,
            subject=request.subject,
            class_level=request.class_level,
            term=request.term,
            metrics=dummy_metrics,
        )

        # Return the result, which will be either a
        # LessonWeek (status: "success") or
        # WeekGenerationError (status: "failed")
        return result

    except Exception as e:
        logger.exception(
            f"Error in retry_lesson_week endpoint for topic: {request.topic}"
        )
        # Return a standard error response if the endpoint itself fails
        raise HTTPException(
            status_code=500, detail=f"Failed to retry week generation: {str(e)}"
        )

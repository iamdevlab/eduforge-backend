from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List
import logging
from app.services.ai_lesson_plan_generator import generate_lesson_plan
from app.models.lesson_plan_model import LessonPlan, LessonPlanRequest
from app.core.security import get_current_user

router = APIRouter()

# -------------------------
# Logging Configuration
# -------------------------
logger = logging.getLogger("lesson_plan_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# -------------------------
# Response Model
# -------------------------
class LessonPlanResponse(BaseModel):
    plan: LessonPlan

    class Config:
        from_attributes = True


# -------------------------
# Helper: summarize fallbacks and enrichments
# -------------------------
def summarize_fallbacks(weeks) -> dict:
    fallback_count = 0
    enrichment_count = 0

    for w in weeks:
        if getattr(w, "_fallback_used", False):
            fallback_count += 1
        elif "Enrichment / consolidation" in (getattr(w, "topic", "") or ""):
            enrichment_count += 1

    return {
        "total_weeks": len(weeks),
        "fallback_weeks": fallback_count,
        "enrichment_weeks": enrichment_count,
        "ai_generated_weeks": len(weeks) - fallback_count - enrichment_count,
    }


# -------------------------
# Protected Endpoint
# -------------------------
@router.post("/lesson-plan", response_model=LessonPlanResponse, summary="Generate AI-powered lesson plan")
async def create_lesson_plan(
    req: LessonPlanRequest, username: str = Depends(get_current_user)
):
    """
    Generate a complete lesson plan for a subject, class, and term.
    Requires:
    - JWT authentication
    - Subject, class_level, term, resumption_date, topics, etc.
    """

    try:
        # Handle comma-separated topics from frontend
        if isinstance(req.topics, str):
            req.topics = [t.strip() for t in req.topics.split(",") if t.strip()]

        # Limit topics to duration weeks
        req.topics = req.topics[:req.duration_weeks or 10]
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

        # Log generation details
        summary = summarize_fallbacks(plan.weeks)
        logger.info(
            f"Lesson plan generated for {req.subject} ({req.class_level}) by {username}. Summary: {summary}"
        )

        return LessonPlanResponse(plan=plan)

    except Exception as e:
        logger.exception("Failed to generate lesson plan")
        raise HTTPException(status_code=500, detail=f"Lesson plan generation failed: {str(e)}")

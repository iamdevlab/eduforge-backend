# app/api/lesson_plan.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, constr
from typing import List, Optional
from datetime import date
import logging

from app.services.ai_lesson_plan_generator import generate_lesson_plan
from app.models.lesson_plan import LessonPlan
from app.core.security import get_current_user

router = APIRouter()

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("lesson_plan_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# -------------------------
# Request and Response Models
# -------------------------
class LessonPlanRequest(BaseModel):
    school_name: constr(strip_whitespace=True)
    state: constr(strip_whitespace=True)
    lga: Optional[constr(strip_whitespace=True)] = None
    subject: constr(strip_whitespace=True)
    class_level: constr(strip_whitespace=True)
    term: constr(strip_whitespace=True)
    resumption_date: date
    duration_weeks: Optional[int] = Field(default=10, ge=1, le=10)
    topics: Optional[List[constr(strip_whitespace=True)]] = None

    def trim_topics(self):
        """
        Trim the topics list to match duration_weeks if it is longer.
        """
        if self.topics and len(self.topics) > self.duration_weeks:
            self.topics = self.topics[:self.duration_weeks]


class LessonPlanResponse(BaseModel):
    plan: LessonPlan


# -------------------------
# Helper: summarize fallback and enrichment
# -------------------------
def summarize_fallbacks(weeks) -> dict:
    """
    Count weeks using AI fallback skeleton and enrichment placeholders.
    Returns a summary dictionary.
    """
    fallback_count = 0
    enrichment_count = 0

    for w in weeks:
        if hasattr(w, "_fallback_used") and w._fallback_used:
            fallback_count += 1
        elif "Enrichment / consolidation" in (w.topic or ""):
            enrichment_count += 1

    return {
        "total_weeks": len(weeks),
        "fallback_weeks": fallback_count,
        "enrichment_weeks": enrichment_count,
        "ai_generated_weeks": len(weeks) - fallback_count - enrichment_count,
    }


# -------------------------
# JWT-Protected Endpoint
# -------------------------
@router.post("/lesson-plan", response_model=LessonPlanResponse)
async def create_lesson_plan(req: LessonPlanRequest, username: str = Depends(get_current_user)):
    """
    Generate a full lesson plan using AI.
    Only accessible to logged-in users.
    Returns weeks with:
        - Objectives
        - Instructional Materials
        - Activities (intro, explanation, guided/independent practice, practical)
        - Assessment
        - Assignment (short questions)
        - Summary (teacher recap)
        - Board Summary (student reference, full explanation)
        - Possible Difficulties
        - Period and Duration
    """
    try:
        req.trim_topics()  # ensure topics list matches duration_weeks

        plan = await generate_lesson_plan(
            school_name=req.school_name,
            state=req.state,
            lga=req.lga,
            subject=req.subject,
            class_level=req.class_level,
            term=req.term,
            resumption_date=req.resumption_date,
            duration_weeks=req.duration_weeks,
            topics=req.topics,
        )

        # Log fallback / enrichment topics
        fallback_summary = summarize_fallbacks(plan.weeks)
        logger.info(
            f"Lesson plan generated for {req.subject} ({req.class_level}) by user {username}. "
            f"Summary: {fallback_summary}"
        )

        return LessonPlanResponse(plan=plan)

    except Exception as e:
        logger.exception("Failed to generate lesson plan")
        raise HTTPException(status_code=500, detail=str(e))

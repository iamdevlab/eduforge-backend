# app/api/routes_questions.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Literal
from app.services.ai_generator import generate_exam_questions

# --- 1. IMPORTS TO ADD/CHANGE ---
from sqlalchemy.orm import Session
from app.models.users import User
from app.subscription_model import SubscriptionTier
from app.services.database import get_db
from app.core.dependencies import get_current_db_user, check_exam_question_limit
# ------------------------------

router = APIRouter()


# --- Request Model ---
class QuestionRequest(BaseModel):  #
    region: str
    subject: str
    class_level: str
    topics: List[str] = Field(..., min_items=1)
    difficulty: str
    num_objectives: int = Field(0, ge=0)
    num_essays: int = Field(0, ge=0)
    essay_style: str = Field(
        default="single",
        description="Essay format: 'single' for flat questions, 'nested' for 1a,1b,1c style",
        pattern="^(single|nested)$",
    )


# --- Response Models ---
class Objective(BaseModel):  #
    question: str
    options: dict = {}


class Essay(BaseModel):  #
    question: str
    sub_questions: List[str] = []  # always a list, never None


class Answer(BaseModel):  #
    type: Literal["objective", "essay"]  # "objective" or "essay"
    answer: str


class QuestionResponse(BaseModel):  #
    objectives: List[Objective]
    essays: List[Essay] = []  # always a list
    answers: List[Answer]


# --- JWT-Protected Endpoint ---
@router.post(
    "/generate",
    response_model=QuestionResponse,
    # --- 2. ADD THE DEPENDENCY ---
    dependencies=[Depends(check_exam_question_limit)],
)
def generate_questions(
    req: QuestionRequest,
    # --- 3. UPDATE THE FUNCTION SIGNATURE ---
    user: User = Depends(get_current_db_user),  # Get the full User object
    db: Session = Depends(get_db),  # Get the DB session
):
    """
    Generates exam questions for a region/subject/class level.
    """
    try:
        output = generate_exam_questions(  #
            region=req.region,
            subject=req.subject,
            class_level=req.class_level,
            topics=req.topics,
            difficulty=req.difficulty,
            num_objectives=req.num_objectives,
            num_essays=req.num_essays,
            essay_style=req.essay_style,
        )

        # --- 4. CRITICAL: INCREMENT THE COUNT ---
        if user.subscription_tier == SubscriptionTier.FREE:
            user.usage.exam_questions_generated += 1
            db.commit()
        # ----------------------------------------

        return output

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Region {req.region} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

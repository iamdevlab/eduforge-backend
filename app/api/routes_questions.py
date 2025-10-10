from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Literal
from app.services.ai_generator import generate_exam_questions
from app.core.security import get_current_user

router = APIRouter()

# --- Request Model ---
class QuestionRequest(BaseModel):
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
        pattern="^(single|nested)$"
    )


# --- Response Models ---
class Objective(BaseModel):
    question: str
    options: dict = {}

class Essay(BaseModel):
    question: str
    sub_questions: List[str] = []  # always a list, never None

class Answer(BaseModel):
    type: Literal["objective", "essay"]  # "objective" or "essay"
    answer: str

class QuestionResponse(BaseModel):
    objectives: List[Objective]
    essays: List[Essay] = []  # always a list
    answers: List[Answer]


# --- JWT-Protected Endpoint ---
@router.post("/generate", response_model=QuestionResponse)
def generate_questions(req: QuestionRequest, username: str = Depends(get_current_user)):
    """
    Generates exam questions for a region/subject/class level.
    Relies on ai_generator to handle both single and nested essay styles.
    """
    try:
        output = generate_exam_questions(
            region=req.region,
            subject=req.subject,
            class_level=req.class_level,
            topics=req.topics,
            difficulty=req.difficulty,
            num_objectives=req.num_objectives,
            num_essays=req.num_essays,
            essay_style=req.essay_style
        )
        return output

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Region {req.region} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

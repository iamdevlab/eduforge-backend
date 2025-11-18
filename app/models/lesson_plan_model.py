# app/models/lesson_plan_model.py
from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field, validator
from datetime import date
import re

# --- HELPER: Text Cleanup (Moved here for automatic validation) ---
def cleanup_text(v: str) -> str:
    if not v:
        return v
    # Remove bolding (**), italics (*), and headings (#)
    v = re.sub(r"\*\*(.*?)\*\*", r"\1", v)
    v = re.sub(r"\*(.*?)\*", r"\1", v)
    v = re.sub(r"#+\s*", "", v)
    # Standardize list-like lines
    v = re.sub(r"^\s*[-*]\s+", "", v, flags=re.MULTILINE)
    return v.strip()

# --- 1. AI CONTENT MODEL (Strictly for AI Generation) ---
class LessonActivity(BaseModel):
    introduction: str
    explanation: str
    guided_practice: Optional[str] = "Teacher guides students through examples."
    independent_practice: Optional[str] = "Students work on exercises individually."
    practical: Optional[str] = "Students perform a related activity."

    @validator("*", pre=True)
    def clean_fields(cls, v):
        if isinstance(v, str):
            return cleanup_text(v)
        return v

class LessonContentAI(BaseModel):
    """
    This model represents ONLY the data the AI generates.
    It excludes system fields like dates, IDs, and status.
    """
    subtopic: Optional[str] = None
    objectives: List[str]
    instructional_materials: List[str]
    prerequisite_knowledge: Optional[str] = None
    activities: LessonActivity
    assessment: Optional[Union[str, List[str]]] = Field(
        default=None, description="Exam questions"
    )
    assignment: Optional[Union[str, List[str]]] = Field(
        default=None, description="Homework"
    )
    summary: str = Field(..., description="Comprehensive 500-800 word lesson note")
    possible_difficulties: Optional[str] = None
    remarks: Optional[str] = None
    period: Optional[str] = "Single"
    duration_minutes: Optional[int] = 40

    # Automatic cleanup for string fields
    @validator("subtopic", "summary", "possible_difficulties", "remarks", "period", pre=True)
    def clean_strings(cls, v):
        if isinstance(v, str):
            return cleanup_text(v)
        return v

    @validator("objectives", "instructional_materials", pre=True)
    def clean_lists(cls, v):
        if isinstance(v, list):
            return [cleanup_text(str(item)) for item in v]
        return v
    
    @validator("assessment", "assignment", pre=True)
    def normalize_lists_to_string(cls, v):
        # AI sometimes returns lists, sometimes strings. We normalize to string for the DB.
        if isinstance(v, list):
            return " ".join([cleanup_text(str(item)) for item in v])
        if isinstance(v, str):
            return cleanup_text(v)
        return v


# --- 2. FULL APP MODEL (Includes Dates & Status) ---
class LessonWeek(BaseModel):
    status: Literal["success"] = "success"
    week_number: int
    start_date: date
    end_date: date
    topic: str
    
    # Embed the content fields
    subtopic: Optional[str] = None
    objectives: List[str]
    instructional_materials: List[str]
    prerequisite_knowledge: Optional[str] = None
    activities: LessonActivity
    assessment: Optional[str]
    assignment: Optional[str]
    summary: str
    possible_difficulties: Optional[str] = None
    remarks: Optional[str] = None
    period: Optional[str] = "Single"
    duration_minutes: Optional[int] = 40

    class Config:
        populate_by_name = True

# --- ERROR MODEL ---
class WeekGenerationError(BaseModel):
    status: Literal["failed"] = "failed"
    week_number: int
    topic: str
    error_message: str
    start_date: date
    end_date: date

# --- PLAN WRAPPER ---
class LessonPlan(BaseModel):
    school_name: str
    state: str
    lga: Optional[str]
    subject: str
    class_level: str
    term: str
    academic_session: Optional[str] = None
    resumption_date: date
    duration_weeks: int = 10
    weeks: List[Union[LessonWeek, WeekGenerationError]]

class LessonPlanRequest(BaseModel):
    school_name: str
    state: str
    lga: Optional[str]
    subject: str
    class_level: str
    term: str
    resumption_date: date
    duration_weeks: Optional[int] = 10
    topics: Union[List[str], str] # Handle comma-separated string or list


# from typing import List, Optional, Union, Literal
# from pydantic import BaseModel, Field
# from datetime import date


# class LessonActivity(BaseModel):
#     introduction: str
#     explanation: str
#     guided_practice: Optional[str] = None
#     independent_practice: Optional[str] = None
#     practical: Optional[str] = None


# class LessonWeek(BaseModel):
#     status: Literal["success"] = "success"

#     week_number: int
#     start_date: date
#     end_date: date
#     topic: str
#     subtopic: Optional[str] = None
#     objectives: List[str]
#     instructional_materials: List[str]
#     prerequisite_knowledge: Optional[str] = None
#     activities: LessonActivity
#     assessment: Optional[str] = Field(
#         default=None, description="Exam-style questions or exercises for the topic."
#     )
#     assignment: Optional[str] = Field(
#         default=None, description="Homework or take-home project for students."
#     )
#     summary: str = Field(
#         ...,
#         description=(
#             "Comprehensive 1000+ word lesson note that serves as both the board summary "
#             "and the main study note. Should include definitions, examples, applications, "
#             "and exam-relevant key points."
#         ),
#     )
#     possible_difficulties: Optional[str] = None
#     remarks: Optional[str] = None
#     period: Optional[str] = "Single"
#     duration_minutes: Optional[int] = 40

#     class Config:
#         populate_by_name = True
#         validate_by_name = True


# #  NEW MODEL TO REPRESENT FAILURE ---
# class WeekGenerationError(BaseModel):
#     status: Literal["failed"] = "failed"
#     week_number: int
#     topic: str
#     error_message: str
#     start_date: date
#     end_date: date


# class LessonPlan(BaseModel):
#     school_name: str
#     state: str
#     lga: Optional[str]
#     subject: str
#     class_level: str  # e.g., "Primary 4", "SS1"
#     term: str  # "First Term", "Second Term", "Third Term"
#     academic_session: Optional[str] = None  # e.g., "2025/2026"
#     resumption_date: date
#     duration_weeks: int = 10  # typically capped at 10 weeks

#     # --- UPDATED WEEKS LIST TO ALLOW FOR ERRORS ---
#     weeks: List[Union[LessonWeek, WeekGenerationError]]


# class LessonPlanRequest(BaseModel):
#     school_name: str
#     state: str
#     lga: Optional[str]
#     subject: str
#     class_level: str
#     term: str
#     resumption_date: date
#     duration_weeks: Optional[int] = 10
#     topics: List[str]

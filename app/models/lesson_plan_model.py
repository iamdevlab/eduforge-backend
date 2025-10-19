from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import date


class LessonActivity(BaseModel):
    introduction: str
    explanation: str
    guided_practice: Optional[str] = None
    independent_practice: Optional[str] = None
    practical: Optional[str] = None


class LessonWeek(BaseModel):
    week_number: int
    start_date: date
    end_date: date
    topic: str
    subtopic: Optional[str] = None
    objectives: List[str]
    instructional_materials: List[str]
    prerequisite_knowledge: Optional[str] = None
    activities: LessonActivity
    assessment: Optional[str] = Field(
        default=None, description="Exam-style questions or exercises for the topic."
    )
    assignment: Optional[str] = Field(
        default=None, description="Homework or take-home project for students."
    )
    summary: str = Field(
        ...,
        description=(
            "Comprehensive 1000+ word lesson note that serves as both the board summary "
            "and the main study note. Should include definitions, examples, applications, "
            "and exam-relevant key points."
        ),
    )
    possible_difficulties: Optional[str] = None
    remarks: Optional[str] = None
    period: Optional[str] = "Single"
    duration_minutes: Optional[int] = 40

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class LessonPlan(BaseModel):
    school_name: str
    state: str
    lga: Optional[str]
    subject: str
    class_level: str  # e.g., "Primary 4", "SS1"
    term: str  # "First Term", "Second Term", "Third Term"
    academic_session: Optional[str] = None  # e.g., "2025/2026"
    resumption_date: date
    duration_weeks: int = 10  # typically capped at 10 weeks
    weeks: List[LessonWeek]


class LessonPlanRequest(BaseModel):
    school_name: str
    state: str
    lga: Optional[str]
    subject: str
    class_level: str
    term: str
    resumption_date: date
    duration_weeks: Optional[int] = 10
    topics: List[str]

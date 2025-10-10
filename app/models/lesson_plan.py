# models/lesson_plan.py
from typing import List, Optional
from pydantic import BaseModel
from datetime import date


class LessonActivity(BaseModel):
    introduction: str
    explanation: str
    guided_practice: Optional[str]
    independent_practice: Optional[str]
    practical: Optional[str]


class LessonWeek(BaseModel):
    week_number: int
    start_date: date
    end_date: date
    topic: str
    subtopic: Optional[str]
    objectives: List[str]
    instructional_materials: List[str]
    prerequisite_knowledge: Optional[str]
    activities: LessonActivity
    assessment: str
    assignment: Optional[str]
    summary: str
    board_summary: Optional[str]  # extra hook for teachers who want to explicitly save board notes
    possible_difficulties: Optional[str]
    remarks: Optional[str]
    period: Optional[str]  # e.g., "Period 1", "Period 2"
    duration_minutes: Optional[int]  # default 40–45 mins in Nigeria


class LessonPlan(BaseModel):
    school_name: str
    state: str
    lga: Optional[str]
    subject: str
    class_level: str  # e.g., "Primary 4", "SS1"
    term: str  # "First Term", "Second Term", "Third Term"
    academic_session: Optional[str]  # e.g., "2025/2026"
    resumption_date: date
    duration_weeks: int = 10  # cap teaching to 10 weeks
    weeks: List[LessonWeek]

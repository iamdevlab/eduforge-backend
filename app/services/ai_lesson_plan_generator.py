import asyncio
import json
import logging
import os
import re
import time
from datetime import date
from functools import lru_cache
from typing import List, Optional, Callable
from asyncio import Semaphore

from app.models.lesson_plan_model import LessonActivity, LessonPlan, LessonWeek
from app.services.ministry_data_service import ministry_service
from app.utils.ai_client import AIClientError, call_ai_model
from app.utils.date_utils import generate_weeks
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field, validator, ValidationError


# -------------------------
# Configuration Management
# -------------------------
class AIConfig(BaseSettings):
    """AI Configuration with environment variable support"""

    api_url: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    api_key: str
    model: str = "gemini-2.5-flash"
    max_retries: int = 2
    timeout: int = 30  # Increased for longer outputs
    max_concurrency: int = 3
    max_duration_weeks: int = 10
    calls_per_second: float = 2.0

    class Config:
        env_prefix = "AI_"
        case_sensitive = False


config = AIConfig()


# -------------------------
# Enhanced Validation Models
# -------------------------
class LessonPlanRequest(BaseModel):
    school_name: str
    subject: str
    class_level: str
    term: str
    resumption_date: date
    duration_weeks: int = Field(default=config.max_duration_weeks, le=config.max_duration_weeks, ge=1)
    topics: Optional[List[str]] = None
    state: Optional[str] = None
    lga: Optional[str] = None

    @validator("subject", "class_level", "term", "state", "lga")
    def normalize_strings(cls, v):
        if v and isinstance(v, str):
            return v.strip().title()
        return v

    @validator("class_level")
    def validate_class_level(cls, v):
        valid_levels = {
            "Basic 1 (Pry 1)", "Basic 2 (Pry 2)", "Basic 3 (Pry 3)",
            "Basic 4 (Pry 4)", "Basic 5 (Pry 5)", "Basic 6 (Pry 6)",
            "Basic 7 (Jss 1)", "Basic 8 (Jss 2)", "Basic 9 (Jss 3)",
            "Ss1", "Ss2", "Ss3"
        }
        if v.title() not in valid_levels:
            raise ValueError(f"Class level must be one of {valid_levels}")
        return v.title()


# -------------------------
# Rate Limiting
# -------------------------
class RateLimiter:
    def __init__(self, calls_per_second: float = config.calls_per_second):
        self.calls_per_second = calls_per_second
        self.semaphore = Semaphore(1)
        self.last_call = 0.0

    async def __aenter__(self):
        async with self.semaphore:
            now = time.time()
            elapsed = now - self.last_call
            wait_time = max(0.0, (1.0 / self.calls_per_second) - elapsed)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_call = time.time()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


# -------------------------
# Caching for Ministry Data
# -------------------------
class CachedMinistryService:
    def __init__(self):
        self._cache = {}
        self._locks = {}

    async def get_ministry_scheme(self, subject: str, class_level: str, term: str, state: Optional[str] = None):
        cache_key = f"{subject}_{class_level}_{term}_{state}"
        if cache_key not in self._cache:
            if cache_key not in self._locks:
                self._locks[cache_key] = asyncio.Lock()
            async with self._locks[cache_key]:
                if cache_key not in self._cache:
                    scheme = await ministry_service.get_ministry_scheme(subject, class_level, term, state)
                    self._cache[cache_key] = scheme
        return self._cache[cache_key]


cached_ministry_service = CachedMinistryService()


# -------------------------
# Logging and Metrics
# -------------------------
logger = logging.getLogger("lesson_plan_generator")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class GenerationMetrics:
    def __init__(self):
        self.start_time = None
        self.successful_weeks = 0
        self.fallback_weeks = 0

    def start(self):
        self.start_time = time.time()

    def record_success(self):
        self.successful_weeks += 1

    def record_fallback(self):
        self.fallback_weeks += 1

    def get_metrics(self, total_weeks: int) -> dict:
        duration = (time.time() - self.start_time) if self.start_time else 0
        success_rate = (self.successful_weeks / total_weeks * 100) if total_weeks else 0
        return {
            "generation_time_seconds": round(duration, 2),
            "successful_weeks": self.successful_weeks,
            "fallback_weeks": self.fallback_weeks,
            "success_rate": f"{success_rate:.1f}%"
        }


# -------------------------
# Retry Decorator
# -------------------------
def retry_on_ai_error(max_retries: int = config.max_retries):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract details for logging and fallback creation
            week_idx = args[0]
            week_meta = args[1]
            topic = args[2]
            metrics = kwargs.get('metrics') or args[6]

            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (AIClientError, ValidationError, json.JSONDecodeError) as e:
                    last_exception = e
                    wait = 2 ** attempt
                    logger.warning(f"Attempt {attempt+1}/{max_retries+1} for week {week_idx} ('{topic}') failed: {e}. Retrying in {wait}s...")
                    await asyncio.sleep(wait)

            logger.error(f"All retries failed for week {week_idx} ('{topic}'). Using fallback. Last error: {last_exception}")
            skeleton = _safe_skeleton_for_topic(topic)
            fallback_payload = {**week_meta, "week_number": week_idx, "topic": topic, **skeleton}
            week_obj = LessonWeek.model_validate(fallback_payload)
            week_obj._fallback_used = True
            metrics.record_fallback()
            return week_obj
        return wrapper
    return decorator

# -------------------------
# NEW: Text Cleanup Utility
# -------------------------
def _cleanup_ai_text(text: str) -> str:
    """Removes common Markdown artifacts from AI-generated text."""
    if not isinstance(text, str):
        return text
    # Remove bolding (**) and italics (*)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Remove markdown headings (###, ##, #)
    text = re.sub(r'#+\s*', '', text)
    # Standardize list-like lines into simple paragraphs
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    return text.strip()

# -------------------------
# Prompt & Fallback Builder
# -------------------------
def _build_enhanced_prompt(subject, class_level, term, week_number, topic):
    # --- MODIFIED PROMPT ---
    return f"""
You are an experienced Nigerian teacher creating a *complete, exam-ready lesson plan* for students.

Your lesson must follow the **Nigerian National Curriculum** and include detailed, realistic, and classroom-usable content.
Return **only a valid JSON object**—no explanations, commentary, or markdown.

Each section must be richly detailed and original.
The "summary" field must contain **at least 1000 words** of well-organized, exam-focused lesson notes. It must include **textually described, well-labeled diagrams** where appropriate for the topic. For example, for a biology topic, describe the diagram of a cell with labeled parts.

The "activities" section should be concise and direct. Each activity (introduction, explanation, etc.) should be a **short, 2-3 sentence summary** of the teacher's actions.

Generate the JSON for:
Subject: {subject}
Class Level: {class_level}
Term: {term}
Week {week_number}
Topic: {topic}

Return strictly valid JSON using this structure:

{{
  "subtopic": "string or null",
  "objectives": [
    "List 3–5 clear and measurable learning objectives."
  ],
  "instructional_materials": [
    "List all materials, resources, and tools required for this lesson."
  ],
  "prerequisite_knowledge": "Describe what students should already know.",
  "activities": {{
    "introduction": "Concise 2-3 sentence summary of how to introduce the topic.",
    "explanation": "Concise 2-3 sentence summary of the core concepts to explain.",
    "guided_practice": "Concise 2-3 sentence summary of teacher-guided exercises.",
    "independent_practice": "Concise 2-3 sentence summary of what students will do individually.",
    "practical": "Concise 2-3 sentence summary of a hands-on class activity."
  }},
  "assessment": [
    "Provide 3–5 realistic exam-style questions."
  ],
  "assignment": [
    "Provide 1–2 take-home assignments or exercises."
  ],
  "summary": "A detailed (1000+ word) comprehensive lesson note. Include all relevant theory, definitions, examples, and applications. Crucially, include text-based diagrams where relevant (e.g., 'DIAGRAM: A Labeled Cross-Section of a Tree Trunk. 1. Bark - The protective outer layer. 2. Sapwood - The living layer that transports water...'). The summary must be suitable as an exam study note.",
  "possible_difficulties": "Predict learning difficulties or misconceptions.",
  "remarks": "Include the teacher’s reflective advice or strategies.",
  "period": "Single or Double",
  "duration_minutes": 40 or 80
}}
"""


def _safe_skeleton_for_topic(topic: str) -> dict:
    # ... (This function remains unchanged)
    return {
        "subtopic": None,
        "objectives": [
            f"Define and explain {topic}.",
            f"List examples of {topic} in everyday life."
        ],
        "instructional_materials": ["Textbook", "Board", "Marker"],
        "prerequisite_knowledge": f"Basic knowledge of previous lessons on {topic}.",
        "activities": {
            "introduction": f"Introduce {topic} through relatable classroom examples.",
            "explanation": f"Explain the meaning, uses, and examples of {topic} clearly.",
            "guided_practice": f"Guide students to identify examples of {topic} in their environment.",
            "independent_practice": f"Students write or discuss how {topic} applies to their daily life.",
            "practical": f"Students demonstrate or simulate real-life examples of {topic}.",
        },
        "assessment": f"1. What is {topic}? 2. Mention examples of {topic}. 3. State two uses of {topic}.",
        "assignment": f"Write short notes on {topic}. Include examples and applications.",
        "summary": f"This lesson covers the meaning, importance, and real-life applications of {topic}. Students should study the definitions, examples, and uses for exam preparation.",
        "possible_difficulties": f"Some students may confuse {topic} with related concepts.",
        "remarks": f"Use visuals or demonstrations to clarify {topic}.",
        "period": "Single",
        "duration_minutes": 40,
    }


# -------------------------
# Main AI Generation Logic
# -------------------------
@retry_on_ai_error(max_retries=config.max_retries)
async def _generate_week_entry(week_idx, week_meta, topic, subject, class_level, term, metrics):
    prompt = _build_enhanced_prompt(subject, class_level, term, week_idx, topic)

    def stringify(value):
        if isinstance(value, list):
            # Apply cleanup to each item in the list before joining
            cleaned_list = [_cleanup_ai_text(str(v)) for v in value]
            return " ".join(cleaned_list)
        if isinstance(value, str):
            return _cleanup_ai_text(value)
        return ""

    raw_output = await call_ai_model(
        prompt,
        api_url=config.api_url,
        api_key=config.api_key,
        model=config.model,
        schema_parser=None,
        max_retries=0,
    )

    if isinstance(raw_output, str):
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.warning(f"AI returned non-JSON string for '{topic}': {e}")
            raise
    else:
        parsed = raw_output

    if not isinstance(parsed, dict):
        raise ValidationError("AI output is not a valid JSON object")

    raw_activities = parsed.get("activities") or {}
    # --- MODIFIED: Apply cleanup to each activity description ---
    activities = {
        "introduction": _cleanup_ai_text(raw_activities.get("introduction") or f"Introduce {topic} and engage students."),
        "explanation": _cleanup_ai_text(raw_activities.get("explanation") or f"Explain {topic} using examples."),
        "guided_practice": _cleanup_ai_text(raw_activities.get("guided_practice") or f"Guide students through {topic} exercises."),
        "independent_practice": _cleanup_ai_text(raw_activities.get("independent_practice") or f"Students practice {topic} individually."),
        "practical": _cleanup_ai_text(raw_activities.get("practical") or f"Demonstrate practical applications of {topic}."),
    }

    week_payload = {
        "week_number": week_idx,
        "start_date": week_meta["start_date"],
        "end_date": week_meta["end_date"],
        "topic": topic,
        # --- MODIFIED: Apply cleanup to all relevant text fields ---
        "subtopic": _cleanup_ai_text(parsed.get("subtopic")),
        "objectives": [_cleanup_ai_text(obj) for obj in parsed.get("objectives", [])] or [f"Explain {topic}."],
        "instructional_materials": [_cleanup_ai_text(mat) for mat in parsed.get("instructional_materials", [])] or ["Textbook", "Board"],
        "prerequisite_knowledge": _cleanup_ai_text(parsed.get("prerequisite_knowledge") or ""),
        "activities": activities,
        "assessment": stringify(parsed.get("assessment")),
        "assignment": stringify(parsed.get("assignment")),
        "summary": _cleanup_ai_text(parsed.get("summary") or f"Study notes on {topic}, including meaning, examples, and uses."),
        "possible_difficulties": _cleanup_ai_text(parsed.get("possible_difficulties") or ""),
        "remarks": _cleanup_ai_text(parsed.get("remarks") or ""),
        "period": _cleanup_ai_text(parsed.get("period") or "Single"),
        "duration_minutes": parsed.get("duration_minutes") or 40,
    }

    week_obj = LessonWeek.model_validate(week_payload)
    metrics.record_success()
    return week_obj

# def _build_enhanced_prompt(subject, class_level, term, week_number, topic):
#     return f"""
# You are an experienced Nigerian teacher creating a *complete, exam-ready lesson plan* for students.

# Your lesson must follow the **Nigerian National Curriculum** and include detailed, realistic, and classroom-usable content.
# Return **only a valid JSON object**—no explanations, commentary, or markdown.

# Each section must be richly detailed and original, suitable for classroom delivery.
# The "summary" field should contain **at least 1000 words** of well-organized, exam-focused lesson notes written in clear, student-friendly English. 
# It should read like a full reference note a teacher gives students to study for tests or exams, covering definitions, explanations, examples, and applications.

# The "activities" section must describe real classroom engagement — teacher–student dialogue, demonstrations, and examples. 
# The "assessment" section must provide exam-style questions aligned with the topic, and "assignment" must specify homework or projects that reinforce the learning objectives.

# Generate the JSON for:
# Subject: {subject}
# Class Level: {class_level}
# Term: {term}
# Week {week_number}
# Topic: {topic}

# Return strictly valid JSON using this structure:

# {{
#   "subtopic": "string or null",
#   "objectives": [
#     "List 3–5 clear and measurable learning objectives."
#   ],
#   "instructional_materials": [
#     "List all materials, resources, and tools required for this lesson."
#   ],
#   "prerequisite_knowledge": "Describe what students should already know before this lesson.",
#   "activities": {{
#     "introduction": "Detailed description (150–300 words) of how the topic will be introduced with questions, examples, or real-life connections.",
#     "explanation": "Full narrative (300–600 words) explaining the core concepts, definitions, and examples related to the topic.",
#     "guided_practice": "Detailed description (150–300 words) of teacher-guided exercises or examples.",
#     "independent_practice": "Detailed description (150–300 words) of what students will do individually to demonstrate understanding.",
#     "practical": "Detailed description (150–300 words) of a practical or hands-on class activity related to the topic."
#   }},
#   "assessment": [
#     "Provide 3–5 realistic exam-style or test questions related to the topic."
#   ],
#   "assignment": [
#     "Provide 1–2 take-home assignments, short essays, or exercises that reinforce the topic."
#   ],
#   "summary": "A detailed (1000+ word) comprehensive lesson note covering the full topic and its subtopics. Write it in a clear, student-friendly tone. Include all relevant theory, definitions, examples, diagrams (described textually), uses, and real-world applications. It must be suitable as an exam study note for Nigerian secondary school students.",
#   "possible_difficulties": "Predict learning difficulties or misconceptions students might have with this topic.",
#   "remarks": "Include the teacher’s reflective advice or strategies for addressing learning difficulties.",
#   "period": "Single or Double",
#   "duration_minutes": 40 or 80
# }}
# """


# def _safe_skeleton_for_topic(topic: str) -> dict:
#     return {
#         "subtopic": None,
#         "objectives": [
#             f"Define and explain {topic}.",
#             f"List examples of {topic} in everyday life."
#         ],
#         "instructional_materials": ["Textbook", "Board", "Marker"],
#         "prerequisite_knowledge": f"Basic knowledge of previous lessons on {topic}.",
#         "activities": {
#             "introduction": f"Introduce {topic} through relatable classroom examples.",
#             "explanation": f"Explain the meaning, uses, and examples of {topic} clearly.",
#             "guided_practice": f"Guide students to identify examples of {topic} in their environment.",
#             "independent_practice": f"Students write or discuss how {topic} applies to their daily life.",
#             "practical": f"Students demonstrate or simulate real-life examples of {topic}.",
#         },
#         "assessment": f"1. What is {topic}? 2. Mention examples of {topic}. 3. State two uses of {topic}.",
#         "assignment": f"Write short notes on {topic}. Include examples and applications.",
#         "summary": f"This lesson covers the meaning, importance, and real-life applications of {topic}. Students should study the definitions, examples, and uses for exam preparation.",
#         "possible_difficulties": f"Some students may confuse {topic} with related concepts.",
#         "remarks": f"Use visuals or demonstrations to clarify {topic}.",
#         "period": "Single",
#         "duration_minutes": 40,
#     }


# # -------------------------
# # Main AI Generation Logic
# # -------------------------
# @retry_on_ai_error(max_retries=config.max_retries)
# async def _generate_week_entry(week_idx, week_meta, topic, subject, class_level, term, metrics):
#     prompt = _build_enhanced_prompt(subject, class_level, term, week_idx, topic)

#     def stringify(value):
#         if isinstance(value, list):
#             return " ".join(str(v) for v in value)
#         if isinstance(value, str):
#             return value.strip()
#         return ""

#     raw_output = await call_ai_model(
#         prompt,
#         api_url=config.api_url,
#         api_key=config.api_key,
#         model=config.model,
#         schema_parser=None,
#         max_retries=0,  # Retries are handled by the decorator
#     )

#     if isinstance(raw_output, str):
#         try:
#             parsed = json.loads(raw_output)
#         except json.JSONDecodeError as e:
#             logger.warning(f"AI returned non-JSON string for '{topic}': {e}")
#             raise  # Propagate to decorator for retry/fallback
#     else:
#         parsed = raw_output

#     if not isinstance(parsed, dict):
#         raise ValidationError("AI output is not a valid JSON object")

#     raw_activities = parsed.get("activities") or {}
#     activities = {
#         "introduction": raw_activities.get("introduction") or f"Introduce {topic} and engage students.",
#         "explanation": raw_activities.get("explanation") or f"Explain {topic} using examples.",
#         "guided_practice": raw_activities.get("guided_practice") or f"Guide students through {topic} exercises.",
#         "independent_practice": raw_activities.get("independent_practice") or f"Students practice {topic} individually.",
#         "practical": raw_activities.get("practical") or f"Demonstrate practical applications of {topic}.",
#     }

#     week_payload = {
#         "week_number": week_idx,
#         "start_date": week_meta["start_date"],
#         "end_date": week_meta["end_date"],
#         "topic": topic,
#         "subtopic": parsed.get("subtopic"),
#         "objectives": parsed.get("objectives") or [f"Explain {topic}."],
#         "instructional_materials": parsed.get("instructional_materials") or ["Textbook", "Board"],
#         "prerequisite_knowledge": parsed.get("prerequisite_knowledge") or "",
#         "activities": activities,
#         "assessment": stringify(parsed.get("assessment")),
#         "assignment": stringify(parsed.get("assignment")),
#         "summary": parsed.get("summary") or f"Study notes on {topic}, including meaning, examples, and uses.",
#         "possible_difficulties": parsed.get("possible_difficulties") or "",
#         "remarks": parsed.get("remarks") or "",
#         "period": parsed.get("period") or "Single",
#         "duration_minutes": parsed.get("duration_minutes") or 40,
#     }

#     # This can raise a Pydantic ValidationError, which will be caught by the decorator
#     week_obj = LessonWeek.parse_obj(week_payload)
#     metrics.record_success()
#     return week_obj


# -------------------------
# Full Lesson Plan Generator
# -------------------------
async def generate_lesson_plan(
    *, school_name, state, lga, subject, class_level, term,
    resumption_date, duration_weeks=config.max_duration_weeks,
    topics=None, concurrency=config.max_concurrency,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> LessonPlan:

    request = LessonPlanRequest(
        school_name=school_name,
        subject=subject,
        class_level=class_level,
        term=term,
        resumption_date=resumption_date,
        duration_weeks=duration_weeks,
        topics=topics,
        state=state,
        lga=lga,
    )

    topics = request.topics
    duration_weeks = request.duration_weeks
    subject = request.subject
    class_level = request.class_level

    logger.info(f"Generating {duration_weeks}-week lesson plan for {subject} {class_level}")

    metrics = GenerationMetrics()
    metrics.start()
    weeks_meta = generate_weeks(resumption_date, duration_weeks)

    if not topics:
        scheme = await cached_ministry_service.get_ministry_scheme(subject, class_level, term, state)
        topics = scheme.topics[:duration_weeks]
    elif len(topics) < duration_weeks:
        pad = duration_weeks - len(topics)
        topics += [f"Enrichment and consolidation {i}" for i in range(1, pad + 1)]

    semaphore = asyncio.Semaphore(concurrency)
    rate_limiter = RateLimiter()

    async def gen_wrapper(idx, meta, topic):
        async with rate_limiter, semaphore:
            result = await _generate_week_entry(idx, meta, topic, subject, class_level, term, metrics=metrics)
            if progress_callback:
                progress_callback(idx, duration_weeks)
            return result

    tasks = [gen_wrapper(i + 1, weeks_meta[i], topics[i]) for i in range(duration_weeks)]
    lesson_weeks = await asyncio.gather(*tasks)

    academic_session = f"{resumption_date.year}/{resumption_date.year + 1}"

    plan = LessonPlan(
        school_name=school_name,
        state=state,
        lga=lga,
        subject=subject,
        class_level=class_level,
        term=term,
        academic_session=academic_session,
        resumption_date=resumption_date,
        duration_weeks=duration_weeks,
        weeks=lesson_weeks,
    )

    logger.info(f"Lesson plan completed successfully with metrics: {metrics.get_metrics(duration_weeks)}")
    return plan

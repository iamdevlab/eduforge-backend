# ai_lesson_plan_generator_enhanced.py
import asyncio
import logging
import os
import time
from datetime import date
from functools import lru_cache
from typing import List, Optional, Callable
from asyncio import Semaphore

from app.models.lesson_plan import LessonActivity, LessonPlan, LessonWeek
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
    api_url: str = "https://api.openai.com/v1/chat/completions"
    api_key: str
    model: str = "gpt-4o-mini"
    max_retries: int = 2
    timeout: int = 15
    max_concurrency: int = 3
    max_duration_weeks: int = 10
    calls_per_second: float = 2.0

    class Config:
        env_prefix = "AI_"
        case_sensitive = False


# Initialize config
config = AIConfig()


# -------------------------
# Enhanced Validation Models
# -------------------------
class LessonPlanRequest(BaseModel):
    """Validated request model for lesson plan generation"""
    school_name: str
    subject: str
    class_level: str
    term: str
    resumption_date: date
    duration_weeks: int = Field(default=config.max_duration_weeks, le=config.max_duration_weeks, ge=1)
    topics: Optional[List[str]] = None
    state: Optional[str] = None
    lga: Optional[str] = None

    @validator('subject', 'class_level', 'term', 'state', 'lga')
    def normalize_strings(cls, v):
        if v and isinstance(v, str):
            return v.strip().title()
        return v

    @validator('class_level')
    def validate_class_level(cls, v):
        valid_levels = { 'Basic 1 (Pry 1)', 'Basic 2 (Pry 2)', 'Basic 3 (Pry 3)',
                        'Basic 4 (Pry 4)', 'Basic 5 (Pry 5)', 'Basic 6 (Pry 6)',
                        'Basic 7 (JSS 1)', 'Basic 8 (JSS 2)', 'Basic 9 (JSS 3)',
                        'SS1', 'SS2', 'SS3'}
        if v.upper() not in valid_levels:
            raise ValueError(f'Class level must be one of {valid_levels}')
        return v.upper()


# -------------------------
# Rate Limiting
# -------------------------
class RateLimiter:
    """Async rate limiter for API calls"""

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
    """Ministry service with caching to reduce API calls"""

    def __init__(self):
        self._cache = {}
        self._locks = {}

    async def get_ministry_scheme(self, subject: str, class_level: str, term: str, state: Optional[str] = None):
        cache_key = f"{subject}_{class_level}_{term}_{state}"

        if cache_key not in self._cache:
            if cache_key not in self._locks:
                self._locks[cache_key] = asyncio.Lock()

            async with self._locks[cache_key]:
                # Double check after acquiring lock
                if cache_key not in self._cache:
                    scheme = await ministry_service.get_ministry_scheme(
                        subject, class_level, term, state
                    )
                    self._cache[cache_key] = scheme

        return self._cache[cache_key]


# Initialize cached service
cached_ministry_service = CachedMinistryService()


# -------------------------
# Enhanced Logging
# -------------------------
class GenerationMetrics:
    """Track and log generation metrics"""

    def __init__(self):
        self.start_time = None
        self.successful_weeks = 0
        self.fallback_weeks = 0

    def start(self):
        self.start_time = time.time()
        self.successful_weeks = 0
        self.fallback_weeks = 0

    def record_success(self):
        self.successful_weeks += 1

    def record_fallback(self):
        self.fallback_weeks += 1

    def get_metrics(self, total_weeks: int) -> dict:
        if self.start_time is None:
            return {}

        duration = time.time() - self.start_time
        success_rate = (self.successful_weeks / total_weeks) * 100 if total_weeks > 0 else 0

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
    """Retry decorator for AI operations"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (AIClientError, ValidationError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed")
            raise last_exception

        return wrapper

    return decorator


# -------------------------
# Logging Configuration
# -------------------------
logger = logging.getLogger("lesson_plan_generator")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


# -------------------------
# Enhanced Prompt Engineering
# -------------------------
def _build_enhanced_prompt(subject: str, class_level: str, term: str, week_number: int, topic: str) -> str:
    """Enhanced prompt with better context and Nigerian curriculum focus"""

    level_context = {
        "JSS1": "Basic concepts, simple explanations, concrete examples, focus on foundational understanding",
        "JSS2": "Building on fundamentals, slightly more complex concepts, practical applications",
        "JSS3": "Preparing for BECE exams, comprehensive coverage, exam techniques",
        "SS1": "Intermediate level, introducing abstract concepts, problem-solving skills",
        "SS2": "Advanced concepts, WAEC/NECO exam-focused, critical thinking",
        "SS3": "Revision and university preparation, complex problem solving, real-world applications"
    }

    context = level_context.get(class_level, "Age-appropriate content with Nigerian context")

    return f"""
You are an expert Nigerian Computer Science teacher creating lesson plans aligned with the Nigerian Ministry of Education curriculum.

CONTEXT:
- Subject: {subject}
- Class Level: {class_level} - {context}
- Term: {term}
- Week Number: {week_number}
- Topic: {topic}
- Curriculum: Nigerian Ministry of Education

SPECIFIC REQUIREMENTS:
1. Objectives must start with measurable verbs (define, explain, calculate, solve, create, etc.)
2. Board summary should be comprehensive and serve as exam preparation material
3. Activities should be practical, engaging, and suitable for Nigerian classroom context
4. Assessment questions should test understanding of key concepts
5. Use Nigerian context and examples where appropriate
6. Instructional materials should be readily available in typical Nigerian schools
7. Keep all content concise and practical for 40-45 minute lessons

CRITICAL: Return ONLY valid JSON. No markdown, no explanations, no additional text.

JSON STRUCTURE:
{{
  "subtopic": "string or null",
  "objectives": ["list", "of", "measurable", "objectives"],
  "instructional_materials": ["list", "of", "available", "materials"],
  "prerequisite_knowledge": "string or null",
  "activities": {{
    "introduction": "engaging opening activity (1-2 sentences)",
    "explanation": "main teaching content (2-3 sentences)", 
    "guided_practice": "teacher-led practice (1-2 sentences or null)",
    "independent_practice": "student individual work (1-2 sentences or null)",
    "practical": "hands-on activity if applicable (1-2 sentences or null)"
  }},
  "board_summary": "detailed student notes with clear sections for exam preparation",
  "assessment": ["2-3 short evaluation questions"],
  "assignment": ["1-2 practical homework tasks"]
}}

Generate the JSON now for {subject} {class_level}, Week {week_number}: {topic}
"""


def _safe_skeleton_for_topic(topic: str) -> dict:
    """Enhanced fallback skeleton with better structure"""
    return {
        "subtopic": None,
        "objectives": [
            f"Define and explain the key concepts of {topic}.",
            f"Identify practical examples and applications of {topic} in daily life.",
            f"Solve basic problems related to {topic}."
        ],
        "instructional_materials": ["Textbook", "Chalkboard/Whiteboard", "Marker/Chalk", "Illustrative charts"],
        "prerequisite_knowledge": f"Basic understanding of related concepts from previous lessons",
        "activities": {
            "introduction": f"Begin with a thought-provoking question about {topic} to engage students.",
            "explanation": f"Systematically explain {topic} using clear examples and simple language.",
            "guided_practice": f"Work through 2-3 examples together with student participation.",
            "independent_practice": f"Students attempt similar problems individually with teacher support.",
            "practical": f"Demonstrate a practical application of {topic} if applicable."
        },
        "board_summary": f"{topic} - Key Points for Exams:\n\n1. DEFINITION: Clear explanation of {topic}\n2. KEY CONCEPTS: Main ideas and principles\n3. EXAMPLES: Practical applications and instances\n4. IMPORTANCE: Why {topic} matters in computer science\n5. COMMON MISTAKES: What to avoid in exams",
        "assessment": [
            f"Explain what {topic} means in your own words.",
            f"Give two real-life examples where {topic} is applied.",
            f"Solve this basic problem related to {topic}."
        ],
        "assignment": [
            f"Research and write a short note on practical uses of {topic}.",
            f"Create a simple diagram or table summarizing key aspects of {topic}."
        ]
    }


# -------------------------
# Main Generation Function with Enhancements
# -------------------------
@retry_on_ai_error()
async def _generate_week_entry(
        week_idx: int,
        week_meta: dict,
        topic: str,
        subject: str,
        class_level: str,
        term: str,
        metrics: GenerationMetrics,
) -> LessonWeek:
    """
    Generate a LessonWeek for a single topic/week using AI with enhanced error handling.
    """
    prompt = _build_enhanced_prompt(subject, class_level, term, week_idx, topic)

    try:
        parsed = await call_ai_model(
            prompt,
            api_url=config.api_url,
            api_key=config.api_key,
            model=config.model,
            schema_parser=None,
            max_retries=config.max_retries,
            timeout=config.timeout
        )

        week_payload = {
            "week_number": week_idx,
            "start_date": week_meta["start_date"],
            "end_date": week_meta["end_date"],
            "topic": topic,
            "subtopic": parsed.get("subtopic"),
            "objectives": parsed.get("objectives") or [],
            "instructional_materials": parsed.get("instructional_materials") or [],
            "prerequisite_knowledge": parsed.get("prerequisite_knowledge"),
            "activities": parsed.get("activities") or {},
            "board_summary": parsed.get("board_summary") or f"{topic}: comprehensive study notes.",
            "assessment": parsed.get("assessment") or [f"Explain the concept of {topic}."],
            "assignment": parsed.get("assignment") or [f"Research assignment on {topic}"],
        }

        week_obj = LessonWeek.parse_obj(week_payload)

        # Ensure minimum quality standards
        if not week_obj.objectives:
            week_obj.objectives = [f"By the end of the lesson, students will understand {topic}."]

        metrics.record_success()
        return week_obj

    except (AIClientError, ValidationError, Exception) as exc:
        logger.warning("AI generation failed for week %s topic %s, using fallback: %s",
                       week_idx, topic, str(exc))
        skeleton = _safe_skeleton_for_topic(topic)
        week_payload = {**week_meta, "week_number": week_idx, "topic": topic, **skeleton}
        week_obj = LessonWeek.parse_obj(week_payload)
        week_obj._fallback_used = True  # Mark as fallback for metrics
        metrics.record_fallback()
        return week_obj


# -------------------------
# Enhanced Lesson Plan Generator
# -------------------------
async def generate_lesson_plan(
        *,
        school_name: str,
        state: str,
        lga: Optional[str],
        subject: str,
        class_level: str,
        term: str,
        resumption_date: date,
        duration_weeks: int = config.max_duration_weeks,
        topics: Optional[List[str]] = None,
        concurrency: int = config.max_concurrency,
        progress_callback: Optional[Callable[[int, int], None]] = None,
) -> LessonPlan:
    """
    Generate a full LessonPlan for a class using AI with all enhancements.

    Features:
    - Enhanced validation and configuration
    - Rate limiting and caching
    - Progress tracking
    - Comprehensive metrics
    - Better error handling and fallbacks
    """
    # Validate input using enhanced request model
    request_data = LessonPlanRequest(
        school_name=school_name,
        subject=subject,
        class_level=class_level,
        term=term,
        resumption_date=resumption_date,
        duration_weeks=duration_weeks,
        topics=topics,
        state=state,
        lga=lga
    )

    # Use validated data
    subject = request_data.subject
    class_level = request_data.class_level
    term = request_data.term
    duration_weeks = request_data.duration_weeks
    topics = request_data.topics

    logger.info("Starting lesson plan generation for %s - %s (%s weeks)",
                subject, class_level, duration_weeks)

    # Initialize metrics tracking
    metrics = GenerationMetrics()
    metrics.start()

    weeks_meta = generate_weeks(resumption_date, duration_weeks)

    # Get topics from ministry service if not provided
    if topics is None:
        logger.info("Fetching topics from cached Ministry scheme service...")
        scheme = await cached_ministry_service.get_ministry_scheme(
            subject, class_level, term, state=state
        )
        topics = scheme.topics[:duration_weeks]
        logger.info("Using scheme source: %s", scheme.source)
    elif len(topics) < duration_weeks:
        # Pad with enrichment topics
        needed = duration_weeks - len(topics)
        enrichment_topics = [f"Enrichment and consolidation {i}" for i in range(1, needed + 1)]
        topics = topics + enrichment_topics
        logger.info("Added %s enrichment topics", needed)
    else:
        topics = topics[:duration_weeks]

    # Setup concurrency and rate limiting
    semaphore = asyncio.Semaphore(concurrency)
    rate_limiter = RateLimiter()

    async def gen_wrapper(idx: int, meta: dict, topic: str):
        """Wrapper with rate limiting and progress tracking"""
        async with rate_limiter, semaphore:
            result = await _generate_week_entry(idx, meta, topic, subject, class_level, term, metrics)

            # Report progress
            if progress_callback:
                progress_callback(idx, duration_weeks)

            return result

    # Generate all weeks concurrently
    tasks = [
        gen_wrapper(i + 1, weeks_meta[i], topics[i])
        for i in range(duration_weeks)
    ]

    lesson_weeks: List[LessonWeek] = await asyncio.gather(*tasks)

    # Create final lesson plan
    plan = LessonPlan(
        school_name=school_name,
        state=state,
        lga=lga,
        subject=subject,
        class_level=class_level,
        term=term,
        resumption_date=resumption_date,
        duration_weeks=duration_weeks,
        weeks=lesson_weeks,
    )

    # Log comprehensive metrics
    metrics_data = metrics.get_metrics(duration_weeks)
    logger.info(
        "Lesson plan generation completed",
        extra={
            "subject": subject,
            "class_level": class_level,
            "duration_weeks": duration_weeks,
            **metrics_data
        }
    )

    return plan


# -------------------------
# Utility Functions
# -------------------------
async def generate_lesson_plan_from_request(request: LessonPlanRequest) -> LessonPlan:
    """Convenience function to generate from validated request"""
    return await generate_lesson_plan(**request.dict())


def get_generation_statistics(lesson_plan: LessonPlan) -> dict:
    """Get statistics about the generation process"""
    total_weeks = len(lesson_plan.weeks)
    fallback_weeks = sum(1 for week in lesson_plan.weeks if getattr(week, '_fallback_used', False))
    successful_weeks = total_weeks - fallback_weeks

    return {
        "total_weeks": total_weeks,
        "successful_ai_generations": successful_weeks,
        "fallback_generations": fallback_weeks,
        "success_rate": f"{(successful_weeks / total_weeks) * 100:.1f}%" if total_weeks > 0 else "0%"
    }
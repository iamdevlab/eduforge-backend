# utils/ai_client.py
# utils/ai_client.py
import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

# Generic type for Pydantic models
T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AIClientError(Exception):
    pass

class AIOverloadError(AIClientError):
    """Raised when the model is overloaded or rate-limited."""
    pass

# ------------------------------
# 1. CIRCUIT BREAKER PATTERN
# ------------------------------
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"

    def can_execute(self) -> bool:
        if self.state == "OPEN":
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        return True

    def record_success(self):
        if self.state != "CLOSED":
            logger.info("Circuit Breaker recovering - switching to CLOSED")
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning("Circuit Breaker tripped! Switching to OPEN state.")

# Global instance (Note: In multi-worker setups like Uvicorn w/ workers > 1, 
# this state is per-process, not shared. Use Redis for shared state if scaling high.)
circuit_breaker = CircuitBreaker()

# ------------------------------
# 2. RAW FETCH
# ------------------------------
async def _fetch_openai_like(prompt: str, api_url: str, api_key: str, model: str) -> str:
    if not circuit_breaker.can_execute():
        raise AIOverloadError("Circuit breaker OPEN — too many recent failures.")

    async with httpx.AsyncClient(timeout=120.0) as client:
        headers = None
        
        # Configuration for Gemini
        if "generativelanguage.googleapis.com" in api_url:
            url = f"{api_url}?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": 4096,
                    "temperature": 0.7,
                }
            }
        # Configuration for OpenAI / Compatible
        else:
            url = api_url
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

        try:
            resp = await client.post(url, json=payload, headers=headers)

            # Handle Circuit Breaker Triggers
            if resp.status_code in (429, 503):
                circuit_breaker.record_failure()
                raise AIOverloadError(f"AI overloaded/limited: {resp.text}")

            if resp.status_code != 200:
                raise AIClientError(f"AI provider error {resp.status_code}: {resp.text}")

            data = resp.json()
            circuit_breaker.record_success()

            # Extract text based on provider
            if "generativelanguage.googleapis.com" in api_url:
                try:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    raise AIClientError(f"Unexpected Gemini structure: {data}")
            else:
                return data["choices"][0]["message"]["content"]

        except httpx.RequestError as e:
            circuit_breaker.record_failure()
            raise AIClientError(f"Request failed: {str(e)}")

# ------------------------------
# 3. JSON EXTRACTION
# ------------------------------
def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:-3].strip()
    elif text.startswith("```"):
        text = text[3:-3].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None

# ------------------------------
# 4. MAIN FUNCTION (Pydantic Integrated)
# ------------------------------
async def call_ai_model(
    prompt: str,
    *,
    api_url: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    response_model: Optional[Type[T]] = None, # Pydantic support
    max_retries: int = 2,
    backoff_base: float = 2.0,
) -> Any:
    """
    Calls AI. If `response_model` (Pydantic class) is passed, validates and returns that object.
    Otherwise returns a dict.
    """
    last_exc = None

    for attempt in range(1, max_retries + 2):
        try:
            logger.info(f"AI call attempt {attempt}")
            
            raw_text = await _fetch_openai_like(prompt, api_url, api_key, model=model)
            parsed_json = _extract_json_from_text(raw_text)
            
            if parsed_json is None:
                raise AIClientError("AI did not return valid JSON.")

            # Pydantic Validation
            if response_model:
                try:
                    return response_model.model_validate(parsed_json)
                except ValidationError as ve:
                    logger.warning(f"Pydantic validation failed: {ve}")
                    # We raise error to trigger a retry, hoping AI fixes format next time
                    raise AIClientError(f"Structure mismatch: {ve}")

            return parsed_json

        except (AIClientError, AIOverloadError, httpx.RequestError) as exc:
            last_exc = exc
            logger.warning(f"AI call failed on attempt {attempt}: {exc}")

            # Don't retry if Circuit Breaker is OPEN (fail fast)
            if isinstance(exc, AIOverloadError):
                break
                
            if attempt <= max_retries:
                sleep_time = backoff_base * (2 ** (attempt - 1))
                await asyncio.sleep(sleep_time)
            else:
                break

    raise AIClientError(f"AI failed. Last error: {last_exc}")




# import asyncio
# import json
# import logging
# import re
# from typing import Any, Callable, Dict, Optional

# import httpx

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


# class AIClientError(Exception):
#     pass


# async def _fetch_openai_like(prompt: str, api_url: str, api_key: str, model: str) -> str: # FIXED: Return type is now str
#     async with httpx.AsyncClient(timeout=120.0) as client:
#         # Detect provider based on URL
#         if "generativelanguage.googleapis.com" in api_url:
#             url = f"{api_url}?key={api_key}"
#             payload = {"contents": [{"parts": [{"text": prompt}]}]}
#             resp = await client.post(url, json=payload)
#         else:
#             headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#             payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
#             resp = await client.post(api_url, headers=headers, json=payload)

#         if resp.status_code != 200:
#             raise AIClientError(f"AI provider returned status {resp.status_code}: {resp.text}")

#         data = resp.json()

#         # Normalize output for both providers
#         if "generativelanguage.googleapis.com" in api_url:
#             # FIXED: Return the raw text directly, do not try to parse or wrap it.
#             return data["candidates"][0]["content"]["parts"][0]["text"]
#         else:
#             return data["choices"][0]["message"]["content"]


# def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
#     """
#     Try to extract a JSON object from a text blob.
#     First, attempt to parse the whole string. If that fails, locate the first {...} block.
#     """
#     text = text.strip()
#     # Handle markdown code blocks
#     if text.startswith("```json"):
#         text = text[7:-3].strip()
#     elif text.startswith("```"):
#         text = text[3:-3].strip()
        
#     try:
#         return json.loads(text)
#     except json.JSONDecodeError:
#         # Fallback to regex for more complex cases if needed, but the above is often sufficient.
#         matches = re.search(r"\{.*\}", text, re.DOTALL)
#         if matches:
#             try:
#                 return json.loads(matches.group(0))
#             except json.JSONDecodeError:
#                 pass # Failed to parse the extracted block
#     logger.warning("Failed to extract any valid JSON from the AI response.")
#     return None


# async def call_ai_model(
#     prompt: str,
#     *,
#     api_url: str,
#     api_key: str,
#     schema_parser: Optional[Callable[[Dict[str, Any]], Any]] = None,
#     max_retries: int = 2,
#     backoff_base: float = 1.0,
#     model: str = "gpt-4o-mini",
# ) -> Any:
#     """
#     Call the AI provider and return a validated object (if schema_parser provided).
#     """

#     last_exc = None
#     for attempt in range(1, max_retries + 2):
#         try:
#             logger.info("AI call attempt %d", attempt)
#             # FIXED: Now receives raw text as a string
#             raw_text = await _fetch_openai_like(prompt, api_url, api_key, model=model)

#             logger.debug("AI raw response (truncated): %s", raw_text[:1000])

#             # FIXED: This logic now correctly executes and extracts the JSON
#             parsed = _extract_json_from_text(raw_text)
            
#             if parsed is None:
#                 # If extraction fails, it's a critical error that should trigger a retry.
#                 raise AIClientError("Failed to extract valid JSON from the AI's response text.")

#             if schema_parser:
#                 try:
#                     return schema_parser(parsed)
#                 except Exception as e:
#                     logger.warning("Schema parser rejected AI output: %s", e)
#                     raise AIClientError(f"Schema validation failed: {e}")

#             return parsed

#         except (AIClientError, httpx.RequestError) as e:
#             logger.exception("AI call failed on attempt %d: %s", attempt, e)
#             last_exc = e
#             if attempt <= max_retries:
#                 sleep_time = backoff_base * (2 ** (attempt - 1))
#                 await asyncio.sleep(sleep_time)
#             else:
#                 break

#     raise AIClientError(f"AI call failed after retries. Last error: {last_exc}")
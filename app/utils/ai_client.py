# utils/ai_client.py
import asyncio
import json
import logging
import re
from typing import Any, Callable, Dict, Optional

import httpx

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AIClientError(Exception):
    pass


async def _fetch_openai_like(prompt: str, api_url: str, api_key: str, model: str) -> str: # FIXED: Return type is now str
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Detect provider based on URL
        if "generativelanguage.googleapis.com" in api_url:
            url = f"{api_url}?key={api_key}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            resp = await client.post(url, json=payload)
        else:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
            resp = await client.post(api_url, headers=headers, json=payload)

        if resp.status_code != 200:
            raise AIClientError(f"AI provider returned status {resp.status_code}: {resp.text}")

        data = resp.json()

        # Normalize output for both providers
        if "generativelanguage.googleapis.com" in api_url:
            # FIXED: Return the raw text directly, do not try to parse or wrap it.
            return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return data["choices"][0]["message"]["content"]


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to extract a JSON object from a text blob.
    First, attempt to parse the whole string. If that fails, locate the first {...} block.
    """
    text = text.strip()
    # Handle markdown code blocks
    if text.startswith("```json"):
        text = text[7:-3].strip()
    elif text.startswith("```"):
        text = text[3:-3].strip()
        
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback to regex for more complex cases if needed, but the above is often sufficient.
        matches = re.search(r"\{.*\}", text, re.DOTALL)
        if matches:
            try:
                return json.loads(matches.group(0))
            except json.JSONDecodeError:
                pass # Failed to parse the extracted block
    logger.warning("Failed to extract any valid JSON from the AI response.")
    return None


async def call_ai_model(
    prompt: str,
    *,
    api_url: str,
    api_key: str,
    schema_parser: Optional[Callable[[Dict[str, Any]], Any]] = None,
    max_retries: int = 2,
    backoff_base: float = 1.0,
    model: str = "gpt-4o-mini",
) -> Any:
    """
    Call the AI provider and return a validated object (if schema_parser provided).
    """

    last_exc = None
    for attempt in range(1, max_retries + 2):
        try:
            logger.info("AI call attempt %d", attempt)
            # FIXED: Now receives raw text as a string
            raw_text = await _fetch_openai_like(prompt, api_url, api_key, model=model)

            logger.debug("AI raw response (truncated): %s", raw_text[:1000])

            # FIXED: This logic now correctly executes and extracts the JSON
            parsed = _extract_json_from_text(raw_text)
            
            if parsed is None:
                # If extraction fails, it's a critical error that should trigger a retry.
                raise AIClientError("Failed to extract valid JSON from the AI's response text.")

            if schema_parser:
                try:
                    return schema_parser(parsed)
                except Exception as e:
                    logger.warning("Schema parser rejected AI output: %s", e)
                    raise AIClientError(f"Schema validation failed: {e}")

            return parsed

        except (AIClientError, httpx.RequestError) as e:
            logger.exception("AI call failed on attempt %d: %s", attempt, e)
            last_exc = e
            if attempt <= max_retries:
                sleep_time = backoff_base * (2 ** (attempt - 1))
                await asyncio.sleep(sleep_time)
            else:
                break

    raise AIClientError(f"AI call failed after retries. Last error: {last_exc}")
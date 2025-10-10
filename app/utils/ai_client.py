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


async def _fetch_openai_like(
    prompt: str,
    api_url: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    timeout: int = 30,
) -> str:
    """
    Minimal wrapper for calling an OpenAI-like completion endpoint.
    Replace this with your provider's request shape.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800,
        "temperature": 0.2,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(api_url, headers=headers, json=payload)
        if resp.status_code >= 400:
            logger.error("AI provider error %s: %s", resp.status_code, resp.text)
            raise AIClientError(f"AI provider returned status {resp.status_code}: {resp.text}")
        body = resp.json()
        # adapt to the provider response structure:
        # for OpenAI: body["choices"][0]["message"]["content"]
        try:
            content = body["choices"][0]["message"]["content"]
        except Exception:
            # fallback: try a common shape
            if "data" in body and isinstance(body["data"], list):
                content = body["data"][0].get("text") or json.dumps(body)
            else:
                content = json.dumps(body)
        return content


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to extract a JSON object from a text blob.
    First, attempt to parse the whole string. If that fails, locate the first {...} block.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        # find JSON object with regex (greedy braces)
        # This simple heuristic finds the first balanced-ish braces group.
        # For robust usage, consider a streaming JSON parser.
        matches = re.findall(r"(\{(?:[^{}]|(?R))*\})", text)
        for m in matches:
            try:
                return json.loads(m)
            except Exception:
                continue
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
    - prompt: the prompt string.
    - api_url: provider endpoint.
    - api_key: provider key.
    - schema_parser: callable to parse/validate the returned dict (e.g., LessonWeek.parse_obj).
    Returns parsed object (if schema_parser) or raw dict.
    Raises AIClientError on unrecoverable failure.
    """
    last_exc = None
    for attempt in range(1, max_retries + 2):  # attempts = max_retries + 1
        try:
            logger.info("AI call attempt %d", attempt)
            raw_text = await _fetch_openai_like(prompt, api_url, api_key, model=model)
            logger.debug("AI raw response: %s", raw_text[:1000])

            parsed = _extract_json_from_text(raw_text)
            if parsed is None:
                raise AIClientError("AI response did not contain valid JSON")

            # Optional schema validation/parsing
            if schema_parser:
                try:
                    obj = schema_parser(parsed)
                    return obj
                except Exception as e:
                    logger.warning("Schema parser rejected AI output: %s", e)
                    # If parser fails, try to return raw dict for debugging, but continue retrying.
                    last_exc = e
                    raise AIClientError(f"Schema validation failed: {e}")

            return parsed

        except (AIClientError, httpx.RequestError) as e:
            logger.exception("AI call failed on attempt %d: %s", attempt, e)
            last_exc = e
            if attempt <= max_retries:
                sleep_time = backoff_base * (2 ** (attempt - 1))
                await asyncio.sleep(sleep_time)
                continue
            else:
                break

    raise AIClientError(f"AI call failed after retries. Last error: {last_exc}")

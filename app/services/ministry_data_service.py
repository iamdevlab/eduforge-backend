# services/ministry_data_service.py
"""
Ministry Data Service (production-ready)

Responsibilities:
- Provide get_ministry_scheme(subject, class_level, term, state=None) -> returns a dict:
    {
      "topics": ["Week 1 topic", ..., "Week N topic"],
      "source": "local" | "remote:<url>" | "generated_fallback",
      "retrieved_at": ISO8601 timestamp,
      "raw": <original raw object for debugging>
    }
- Prefer local JSON store (authoritative), optionally fetch remote sources and update cache.
- Validate and normalise topic lists, pad to `target_weeks` (default 10).
- Thread-safe and async-friendly; TTL cache to avoid excessive remote calls.
- Clear logging and deterministic fallbacks to avoid AI hallucination sources.

Usage:
    from services.ministry_data_service import ministry_service
    scheme = await ministry_service.get_ministry_scheme("Biology", "SS1", "First Term", state="Lagos")
    topics = scheme["topics"]  # list of strings length == 10 (padded if originally shorter)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, ValidationError, constr



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -------------------------
# Configuration (override from env in production)
# -------------------------
LOCAL_SCHEME_DIR = Path(os.getenv("MINISTRY_SCHEME_DIR", "./data/ministry_schemes"))
REMOTE_SOURCES = os.getenv("MINISTRY_REMOTE_SOURCES", "")  # comma-separated URLs; keep empty if none
REMOTE_SOURCES_LIST = [u.strip() for u in REMOTE_SOURCES.split(",") if u.strip()]
CACHE_TTL_SECONDS = int(os.getenv("MINISTRY_CACHE_TTL", "3600"))  # 1 hour default
HTTP_TIMEOUT_SECONDS = int(os.getenv("MINISTRY_HTTP_TIMEOUT", "10"))
HTTP_MAX_RETRIES = int(os.getenv("MINISTRY_HTTP_MAX_RETRIES", "2"))
TARGET_WEEKS = int(os.getenv("MINISTRY_TARGET_WEEKS", "10"))

# Ensure local dir exists (not strictly required but helpful)
LOCAL_SCHEME_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Models
# -------------------------
class SchemeQuery(BaseModel):
    subject: constr(strip_whitespace=True)
    class_level: constr(strip_whitespace=True)
    term: constr(strip_whitespace=True)
    state: Optional[constr(strip_whitespace=True)] = None


class SchemeResult(BaseModel):
    topics: List[str]  # normalized list of topics for weeks (length == TARGET_WEEKS)
    source: str  # 'local', 'remote:<url>', 'generated_fallback'
    retrieved_at: datetime
    raw: Optional[dict] = None  # original raw payload for debugging


# -------------------------
# Exceptions
# -------------------------
class MinistryDataError(Exception):
    pass


# -------------------------
# Helper functions
# -------------------------
def _normalise_topic(t: Any) -> str:
    """Convert topic-like objects to a clean string."""
    if t is None:
        return ""
    return str(t).strip()


def _pad_topics(topics: List[str], target: int = TARGET_WEEKS) -> List[str]:
    """Pad or truncate topics list to exactly `target` items."""
    cleaned = [_normalise_topic(t) for t in topics if _normalise_topic(t) != ""]
    if len(cleaned) >= target:
        return cleaned[:target]
    # pad with generic placeholders
    for i in range(len(cleaned) + 1, target + 1):
        cleaned.append(f"Enrichment / consolidation week {i}")
    return cleaned


def _local_filename_for(query: SchemeQuery) -> Path:
    """
    Build a deterministic filename for local JSON files.
    e.g. biology_ss1_first-term__lagos.json
    """
    parts = [
        query.subject.lower().replace(" ", "_"),
        query.class_level.lower().replace(" ", "_"),
        query.term.lower().replace(" ", "_"),
    ]
    if query.state:
        parts.append(query.state.lower().replace(" ", "_"))
    filename = "__".join(parts) + ".json"
    return LOCAL_SCHEME_DIR / filename


async def _http_get_with_retries(url: str, timeout: int = HTTP_TIMEOUT_SECONDS, retries: int = HTTP_MAX_RETRIES) -> Tuple[int, str]:
    """Simple HTTP GET with retries and backoff (async). Returns (status_code, text)."""
    backoff = 1.0
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, retries + 2):
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.status_code, resp.text
            except Exception as e:
                logger.warning("HTTP GET failed for %s (attempt %d/%d): %s", url, attempt, retries + 1, e)
                if attempt <= retries:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                raise


def _parse_remote_payload_for_scheme(payload_text: str, query: SchemeQuery) -> Optional[Dict[str, Any]]:
    """
    Try to parse remote JSON/structured text into a canonical dict:
    {
       "topics": [...],
       "metadata": {...}
    }
    The remote source formats vary; preferred is JSON shape matching our keys.
    """
    try:
        obj = json.loads(payload_text)
    except Exception:
        # Not JSON — bail early
        return None

    # Common shapes:
    # 1) Direct mapping: obj.get("subjects", {})[subject][class_level][term] -> list
    # 2) Flat mapping keyed by deterministic filename
    # 3) Simple object: {"topics": [...]}
    # We'll try a few heuristics.
    sub = query.subject.lower()
    cl = query.class_level.lower()
    term = query.term.lower()

    # Direct "topics" key
    if isinstance(obj, dict) and "topics" in obj and isinstance(obj["topics"], list):
        return {"topics": obj["topics"], "raw": obj}

    # Search for nested subject/class/term
    def deep_get(dct: dict, *keys):
        cur = dct
        for k in keys:
            if not isinstance(cur, dict):
                return None
            # try direct, try lower keys
            if k in cur:
                cur = cur[k]
                continue
            found = None
            for kk in cur:
                if isinstance(kk, str) and kk.lower() == k:
                    found = cur[kk]
                    break
            if found is None:
                return None
            cur = found
        return cur

    candidate = deep_get(obj, query.subject, query.class_level, query.term)
    if candidate and isinstance(candidate, list):
        return {"topics": candidate, "raw": obj}

    # Lastly, check key with deterministic filename
    key = f"{query.subject}_{query.class_level}_{query.term}".lower().replace(" ", "_")
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and key in k.lower():
                if isinstance(v, list):
                    return {"topics": v, "raw": obj}
                if isinstance(v, dict) and "topics" in v:
                    return {"topics": v["topics"], "raw": obj}

    return None


# -------------------------
# Service class
# -------------------------
@dataclass
class MinistryDataService:
    """
    Thread-safe async service to provide ministry scheme of work topics.
    Uses in-memory TTL cache and local JSON as primary source.
    """

    _cache: Dict[str, Tuple[SchemeResult, float]] = None  # key -> (SchemeResult, expiry_ts)
    _lock: asyncio.Lock = None

    def __post_init__(self):
        if self._cache is None:
            self._cache = {}
        if self._lock is None:
            self._lock = asyncio.Lock()

    def _cache_key(self, query: SchemeQuery) -> str:
        parts = [query.subject.lower(), query.class_level.lower(), query.term.lower()]
        if query.state:
            parts.append(query.state.lower())
        return "::".join(parts)

    async def _read_local_file(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8")
            return json.loads(text)
        except Exception as e:
            logger.exception("Failed to read/parse local file %s: %s", path, e)
            return None

    async def _try_local(self, query: SchemeQuery) -> Optional[SchemeResult]:
        path = _local_filename_for(query)
        data = await self._read_local_file(path)
        if not data:
            return None
        # heuristics similar to remote parser
        if isinstance(data, dict) and "topics" in data and isinstance(data["topics"], list):
            topics = _pad_topics(data["topics"], TARGET_WEEKS)
            return SchemeResult(
                topics=topics, source=f"local:{path.name}", retrieved_at=datetime.now(timezone.utc), raw=data
            )
        # try to find key inside file
        parsed = _parse_remote_payload_for_scheme(json.dumps(data), query)
        if parsed:
            topics = _pad_topics(parsed["topics"], TARGET_WEEKS)
            return SchemeResult(topics=topics, source=f"local:{path.name}", retrieved_at=datetime.now(timezone.utc), raw=parsed.get("raw"))
        return None

    async def _try_remotes(self, query: SchemeQuery) -> Optional[SchemeResult]:
        # iterate through configured remote sources; stop at first valid result
        for url in REMOTE_SOURCES_LIST:
            try:
                status, text = await _http_get_with_retries(url)
                parsed = _parse_remote_payload_for_scheme(text, query)
                if parsed:
                    topics = _pad_topics(parsed["topics"], TARGET_WEEKS)
                    return SchemeResult(topics=topics, source=f"remote:{url}", retrieved_at=datetime.now(timezone.utc), raw=parsed.get("raw"))
            except Exception as e:
                logger.warning("Remote source %s failed: %s", url, e)
                continue
        return None

    async def get_ministry_scheme(self, subject: str, class_level: str, term: str, state: Optional[str] = None) -> SchemeResult:
        """
        Main entrypoint. Returns a SchemeResult with normalized topics list (len == TARGET_WEEKS)
        and metadata about the source.
        """

        query = SchemeQuery(subject=subject, class_level=class_level, term=term, state=state)
        key = self._cache_key(query)

        async with self._lock:
            # check cache
            entry = self._cache.get(key)
            now_ts = datetime.now(timezone.utc).timestamp()
            if entry:
                scheme_result, expiry = entry
                if now_ts < expiry:
                    logger.debug("Cache hit for %s", key)
                    return scheme_result
                else:
                    logger.debug("Cache expired for %s", key)
                    # drop through to refresh

            # 1) Try local authoritative file
            local = await self._try_local(query)
            if local:
                expiry = now_ts + CACHE_TTL_SECONDS
                self._cache[key] = (local, expiry)
                logger.info("Using local scheme for %s (source=%s)", key, local.source)
                return local

            # 2) Try remote sources (if configured)
            if REMOTE_SOURCES_LIST:
                remote = await self._try_remotes(query)
                if remote:
                    expiry = now_ts + CACHE_TTL_SECONDS
                    self._cache[key] = (remote, expiry)
                    logger.info("Using remote scheme for %s (source=%s)", key, remote.source)
                    return remote

            # 3) Nothing found: return a generated fallback (deterministic placeholders)
            topics = [f"Enrichment / consolidation week {i}" for i in range(1, TARGET_WEEKS + 1)]
            fallback = SchemeResult(topics=topics, source="generated_fallback", retrieved_at=datetime.now(timezone.utc), raw=None)
            expiry = now_ts + CACHE_TTL_SECONDS
            self._cache[key] = (fallback, expiry)
            logger.warning("No scheme found for %s; returning fallback topics", key)
            return fallback


# Module-level single instance (convenience)
ministry_service = MinistryDataService()


# -------------------------
# Convenience sync wrappers for sync code
# -------------------------
def get_ministry_scheme_sync(subject: str, class_level: str, term: str, state: Optional[str] = None) -> SchemeResult:
    """
    Synchronous wrapper for environments that don't want async usage.
    Runs the async get_ministry_scheme in an event loop.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Running inside an existing loop (e.g., FastAPI). Caller should use async.
        raise RuntimeError("Event loop running — call ministry_service.get_ministry_scheme(...) asynchronously instead")

    return asyncio.run(ministry_service.get_ministry_scheme(subject, class_level, term, state))


# -------------------------
# Example CLI quick test (not executed on import)
# -------------------------
if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--class_level", required=True)
    parser.add_argument("--term", required=True)
    parser.add_argument("--state", required=False)
    args = parser.parse_args()

    async def main():
        res = await ministry_service.get_ministry_scheme(args.subject, args.class_level, args.term, args.state)
        print("Source:", res.source)
        print("Retrieved:", res.retrieved_at.isoformat())
        print("Topics (count={}):".format(len(res.topics)))
        for i, t in enumerate(res.topics, start=1):
            print(f" Week {i}: {t}")

    asyncio.run(main())

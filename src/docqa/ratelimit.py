"""Recognise provider quota errors and how long the provider asks to wait."""

from __future__ import annotations

import re

_RETRY_AFTER_RE = re.compile(r"retry in (?:(\d+)h)?(?:(\d+)m(?!s))?(?:([0-9.]+)s)?", re.IGNORECASE)


class QuotaExhaustedError(RuntimeError):
    """The provider quota will not recover soon (e.g. daily limit); retrying now is pointless."""


def is_rate_limited(exc: Exception) -> bool:
    """True for provider quota errors (HTTP 429 / RESOURCE_EXHAUSTED), whatever the SDK."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code == 429 or "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc)[:20]


def retry_delay(exc: Exception, attempt: int) -> float:
    """The wait the provider asked for ("retry in 9.5s", "retry in 14h51m28s"), else backoff."""
    match = _RETRY_AFTER_RE.search(str(exc))
    if match and any(match.groups()):
        hours, minutes, seconds = (float(g) if g else 0.0 for g in match.groups())
        return hours * 3600 + minutes * 60 + seconds + 1.0
    return min(5.0 * 2**attempt, 60.0)

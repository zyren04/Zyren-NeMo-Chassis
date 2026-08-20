"""
Strict No-Burst Rate Limiting
Converts user-specified rate (requests/second) into AsyncLimiter(max_rate=1, time_period=1/rate)
to enforce exactly one request per interval — eliminating burst allowance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiolimiter import AsyncLimiter

if TYPE_CHECKING:
    from .nim_client import RateLimiter as TokenBucketRateLimiter


@dataclass
class StrictRateLimiter:
    """
    No-burst rate limiter: exactly 1 request per (1/rate) seconds.

    Unlike TokenBucket which allows bursts up to capacity, this enforces
    strict spacing between requests. Critical for cloud NIM APIs with
    hard rate limits where bursts trigger 429 errors.
    """

    rate_per_second: float
    _limiter: AsyncLimiter = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        # max_rate=1 with time_period=1/rate enforces exactly one request per interval
        self._limiter = AsyncLimiter(max_rate=1, time_period=1.0 / self.rate_per_second)

    async def acquire(self) -> None:
        """Acquire permission to make a request, waiting if necessary."""
        await self._limiter.acquire()

    def __repr__(self) -> str:
        return f"StrictRateLimiter(rate_per_second={self.rate_per_second})"


class RateLimitMode:
    """Rate limiting mode for NIMClient."""

    TOKEN_BUCKET = "token_bucket"  # nosec: B105 - not a password, just a mode identifier
    STRICT = "strict"  # No bursts, exact spacing


def create_rate_limiter(
    mode: str,
    max_rpm: int,
    max_concurrent: int = 3,
) -> tuple[StrictRateLimiter | None, TokenBucketRateLimiter | None]:
    """
    Factory to create rate limiter based on mode.

    Returns:
        Tuple of (strict_limiter, token_bucket_limiter) - one will be None.
    """
    from .nim_client import RateLimiter as TokenBucketRateLimiter

    if mode == RateLimitMode.STRICT:
        rate_per_second = max_rpm / 60.0
        return StrictRateLimiter(rate_per_second=rate_per_second), None
    else:
        return None, TokenBucketRateLimiter(max_concurrent=max_concurrent, max_rpm=max_rpm)


__all__ = [
    "StrictRateLimiter",
    "RateLimitMode",
    "create_rate_limiter",
]

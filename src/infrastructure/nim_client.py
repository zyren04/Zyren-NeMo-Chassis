"""
NVIDIA NIM Gateway & Rate-Limiter
Centralized wrapper around langchain-nvidia-ai-endpoints with async traffic limiting, retry logic, and auto-continuation.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .rate_limiting import RateLimitMode, StrictRateLimiter

# Load .env file from project root or current directory
_env_paths = [
    Path.cwd() / ".env",
    Path(__file__).parent.parent.parent / ".env",  # project root from src/infrastructure/
]
for _env_path in _env_paths:
    if _env_path.exists():
        load_dotenv(_env_path)
        break


@dataclass
class TokenBucket:
    """Token bucket rate limiter for RPM (Requests Per Minute) control."""

    capacity: int
    refill_rate: float  # tokens per second
    _tokens: float = field(default=0.0, init=False)
    _last_refill: float = field(default=0.0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        self._tokens = float(self.capacity)
        self._last_refill = time.monotonic()

    async def take(self, tokens: int = 1) -> float:
        """
        Take tokens from bucket, returning wait time if needed.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            Wait time in seconds (0.0 if token available immediately).
        """
        # This method is fully typed above
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(float(self.capacity), self._tokens + elapsed * self.refill_rate)
            self._last_refill = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0

            # Calculate wait time required for refill
            needed = tokens - self._tokens
            wait_time = needed / self.refill_rate
            self._tokens = 0.0
            return wait_time


class RateLimiter:
    """
    Composite rate limiter with semaphore for concurrency and token bucket for RPM.
    """

    def __init__(
        self,
        max_concurrent: int = 3,
        max_rpm: int = 40,
    ):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.token_bucket = TokenBucket(
            capacity=max_rpm,
            refill_rate=max_rpm / 60.0,  # tokens per second
        )
        self._lock = asyncio.Lock()
        self._request_count = 0
        self._total_wait_time = 0.0

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        """
        Acquire both semaphore and rate limit tokens.
        Ensures compliance with maximum concurrency and RPM bounds.
        Order: semaphore first (concurrency), then token bucket (RPM).
        This prevents token drift if semaphore blocks.
        """
        # Acquire semaphore for concurrency control FIRST
        async with self.semaphore:
            # Then wait for rate limit token
            wait_time = await self.token_bucket.take(1)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                self._total_wait_time += wait_time

            self._request_count += 1
            yield

    @property
    def stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "request_count": self._request_count,
            "total_wait_time": self._total_wait_time,
            "available_tokens": self.token_bucket._tokens,
            "capacity": self.token_bucket.capacity,
        }


class NIMClient:
    """
    Centralized NVIDIA NIM client with built-in rate limiting, retry logic, and auto-continuation.

    Features:
    - Async semaphore (max 3 concurrent)
    - Token bucket rate limiter (40 RPM) - default mode, allows bursts
    - Strict rate limiter (no bursts, exact spacing) - for cloud NIM APIs
    - Exponential backoff retry with jitter
    - Automatic response continuation for finish_reason == 'length'
    - Reads NVIDIA_API_KEY from environment
    """

    def __init__(
        self,
        model: str = "nvidia/nemotron-3-ultra-550b-a55b",
        api_key: str | None = None,
        base_url: str | None = None,
        max_concurrent: int = 3,
        max_rpm: int = 40,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        timeout: float = 120.0,
        auto_continue_truncated: bool = True,
        max_continuations: int = 3,
        rate_limit_mode: str = RateLimitMode.TOKEN_BUCKET,
        **chat_kwargs: Any,
    ):
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.auto_continue_truncated = auto_continue_truncated
        self.max_continuations = max_continuations
        self.rate_limit_mode = rate_limit_mode

        # Type annotations for rate limiters (initialized conditionally)
        self.strict_rate_limiter: StrictRateLimiter | None = None
        self.rate_limiter: RateLimiter | None = None
        self._semaphore: asyncio.Semaphore | None = None

        # Resolve API key from arguments or environment
        resolved_api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "NVIDIA_API_KEY not provided. Set environment variable or pass api_key parameter."
            )

        # Initialize rate limiter based on mode
        if rate_limit_mode == RateLimitMode.STRICT:
            self.strict_rate_limiter = StrictRateLimiter(rate_per_second=max_rpm / 60.0)
            self.rate_limiter = None
            # Still need semaphore for concurrency control
            self._semaphore = asyncio.Semaphore(max_concurrent)
        else:
            self.strict_rate_limiter = None
            self.rate_limiter = RateLimiter(
                max_concurrent=max_concurrent,
                max_rpm=max_rpm,
            )

        # Initialize underlying ChatNVIDIA client
        self._client = ChatNVIDIA(
            model=model,
            api_key=resolved_api_key,
            base_url=base_url,
            timeout=timeout,
            **chat_kwargs,
        )

    @retry(
        wait=wait_exponential_jitter(initial=1, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, IOError)),
        reraise=True,
    )
    async def _invoke_single_call(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        """
        Execute a single invocation under the rate-limiting context.
        """
        if self.rate_limit_mode == RateLimitMode.STRICT:
            # Strict mode: acquire semaphore for concurrency, then strict rate limiter
            assert self.strict_rate_limiter is not None
            assert self._semaphore is not None
            async with self._semaphore:
                await self.strict_rate_limiter.acquire()
                response = await self._client.ainvoke(messages, **kwargs)
                return response
        else:
            # Token bucket mode: use composite rate limiter
            assert self.rate_limiter is not None
            async with self.rate_limiter.acquire():
                response = await self._client.ainvoke(messages, **kwargs)
                return response

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> AIMessage:
        """
        Invoke the model asynchronously with rate limiting, retries, and automatic continuation.

        Args:
            messages: List of messages to send to the model.
            **kwargs: Additional arguments passed to the underlying client.

        Returns:
            AIMessage with the complete generated content.
        """
        current_messages = list(messages)
        full_response = await self._invoke_single_call(current_messages, **kwargs)

        # Auto-continuation logic to prevent code cutoff
        if self.auto_continue_truncated:
            continuations = 0
            while continuations < self.max_continuations:
                finish_reason = full_response.response_metadata.get(
                    "finish_reason"
                ) or full_response.response_metadata.get("stop_reason")
                if finish_reason == "length":
                    current_messages.append(full_response)
                    current_messages.append(
                        HumanMessage(
                            content="Continue outputting the remaining code from the exact last character without repeating anything."
                        )
                    )
                    next_chunk = await self._invoke_single_call(current_messages, **kwargs)
                    full_response = AIMessage(
                        content=str(full_response.content) + str(next_chunk.content),
                        response_metadata=next_chunk.response_metadata,
                    )
                    continuations += 1
                else:
                    break

        return full_response

    async def abatch(
        self,
        messages_list: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> list[AIMessage]:
        """
        Batch invoke multiple message sequences concurrently.

        Args:
            messages_list: List of message lists.
            **kwargs: Additional arguments passed to the underlying client.

        Returns:
            List of AIMessage responses.
        """
        tasks = [self.ainvoke(messages, **kwargs) for messages in messages_list]
        return await asyncio.gather(*tasks)

    async def astream(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        Stream the model response asynchronously.

        Args:
            messages: List of messages to stream.
            **kwargs: Additional arguments passed to the underlying client.

        Yields:
            ChatGenerationChunk items as they arrive.
        """
        if self.rate_limit_mode == RateLimitMode.STRICT:
            assert self.strict_rate_limiter is not None
            assert self._semaphore is not None
            async with self._semaphore:
                await self.strict_rate_limiter.acquire()
                stream = self._client.astream(messages, **kwargs)
                async for chunk in stream:
                    if isinstance(chunk, ChatGenerationChunk):
                        yield chunk
                    else:
                        yield ChatGenerationChunk(message=chunk)
        else:
            assert self.rate_limiter is not None
            async with self.rate_limiter.acquire():
                stream = self._client.astream(messages, **kwargs)
                async for chunk in stream:
                    if isinstance(chunk, ChatGenerationChunk):
                        yield chunk
                    else:
                        yield ChatGenerationChunk(message=chunk)

    async def ainvoke_simple(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Simple invoke interface with raw string prompt.

        Args:
            prompt: User prompt text.
            system_prompt: Optional system instructions.
            **kwargs: Additional arguments.

        Returns:
            Generated text content as string.
        """
        messages: list[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        result = await self.ainvoke(messages, **kwargs)
        return str(result.content)

    def get_stats(self) -> dict[str, Any]:
        """
        Get client and rate limiter runtime statistics.
        """
        stats = {
            "model": self.model,
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "timeout": self.timeout,
            "rate_limit_mode": self.rate_limit_mode,
        }
        if self.rate_limit_mode == RateLimitMode.STRICT:
            assert self.strict_rate_limiter is not None
            stats["strict_rate_limiter"] = {
                "rate_per_second": self.strict_rate_limiter.rate_per_second,
            }
        else:
            assert self.rate_limiter is not None
            stats["rate_limiter"] = self.rate_limiter.stats
        return stats

    @property
    def client(self) -> ChatNVIDIA:
        """Access underlying ChatNVIDIA client."""
        return self._client


# Global default client instance
_default_client: NIMClient | None = None


def get_nim_client(**kwargs: Any) -> NIMClient:
    """Get or create the default NIM client instance."""
    global _default_client
    if _default_client is None:
        _default_client = NIMClient(**kwargs)
    return _default_client


def set_nim_client(client: NIMClient) -> None:
    """Set the default NIM client instance (useful for testing and dependency injection)."""
    global _default_client
    _default_client = client


__all__ = [
    "NIMClient",
    "RateLimiter",
    "TokenBucket",
    "get_nim_client",
    "set_nim_client",
]

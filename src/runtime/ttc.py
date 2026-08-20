"""
Test-Time Compute (TTC) Framework

Generic test-time compute executor for any async function.
Executes a function N times with configurable selection strategy
(majority voting, best-of-N, etc.) and optional early stopping.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")  # Input type
R = TypeVar("R")  # Output type


@dataclass
class TTCConfig:
    """Configuration for Test-Time Compute execution."""

    num_executions: int = 3
    max_concurrency: int | None = None
    early_stop_threshold: int | bool = False
    selector: Callable[[list[R]], R] = field(
        default_factory=lambda: lambda x: x[0]  # default: first result
    )

    def __post_init__(self) -> None:
        if self.num_executions < 1:
            raise ValueError("num_executions must be >= 1")
        if self.max_concurrency is not None and self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1 or None")
        if isinstance(self.early_stop_threshold, int) and self.early_stop_threshold < 1:
            raise ValueError("early_stop_threshold must be >= 1 or False")


class TTCExecutor(Generic[T, R]):
    """
    Generic test-time compute executor for any async function.

    Executes the function multiple times in parallel (with optional concurrency limit)
    and applies a selection strategy to choose the best result.

    Example:
        async def my_llm_call(prompt: str) -> str:
            return await llm.ainvoke(prompt)

        config = TTCConfig(
            num_executions=5,
            max_concurrency=3,
            early_stop_threshold=3,  # Stop if 3 results agree
            selector=majority_vote_selector,
        )
        executor = TTCExecutor(config, my_llm_call)
        result = await executor.execute("What is 2+2?")
    """

    def __init__(self, config: TTCConfig, func: Callable[[T], Awaitable[R]]) -> None:
        self.config = config
        self.func = func
        self._semaphore: asyncio.Semaphore | None = None

        if config.max_concurrency is not None:
            self._semaphore = asyncio.Semaphore(config.max_concurrency)

    async def _execute_single(self, input: T) -> R:
        """Execute the function once, with optional semaphore."""
        if self._semaphore:
            async with self._semaphore:
                return await self.func(input)
        return await self.func(input)

    async def execute(self, input: T) -> R:
        """
        Execute the function multiple times and select the best result.

        Args:
            input: Input to pass to the function.

        Returns:
            Selected result based on the configured selector.
        """
        # Create tasks for parallel execution
        tasks = [
            asyncio.create_task(self._execute_single(input))
            for _ in range(self.config.num_executions)
        ]

        results: list[R] = []

        # Execute with early stopping
        for completed, coro in enumerate(asyncio.as_completed(tasks), 1):
            result = await coro
            results.append(result)

            # Check early stopping condition
            if self._should_early_stop(results, completed):
                # Cancel remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                # Wait for cancelled tasks to clean up
                await asyncio.gather(*tasks, return_exceptions=True)
                break

        if not results:
            raise RuntimeError("No results produced by TTCExecutor")

        # Apply selection strategy
        return self.config.selector(results)

    def _should_early_stop(self, results: list[R], completed: int) -> bool:
        """Check if early stopping condition is met."""
        threshold = self.config.early_stop_threshold
        if threshold is False:
            return False
        if isinstance(threshold, int):
            # Early stop if we have enough results that agree
            # This requires the selector to be able to determine agreement
            # For now, we just check if we have enough results
            return completed >= threshold
        return False


# Common selector functions
def first_result_selector(results: list[R]) -> R:
    """Select the first result (default)."""
    return results[0]


def majority_vote_selector(results: list[R]) -> R:
    """
    Select the majority vote result.
    Requires results to be hashable and comparable.
    """
    if not results:
        raise ValueError("No results to select from")

    # Count occurrences
    counts: dict[R, int] = {}
    for r in results:
        counts[r] = counts.get(r, 0) + 1

    # Return the most common
    return max(counts.items(), key=lambda x: x[1])[0]


def best_of_n_selector(
    results: list[R],
    scorer: Callable[[R], float],
) -> R:
    """
    Select the result with the highest score.

    Args:
        results: List of results to score.
        scorer: Function that returns a score for a result (higher is better).

    Returns:
        Result with the highest score.
    """
    if not results:
        raise ValueError("No results to select from")
    return max(results, key=scorer)


__all__ = [
    "TTCConfig",
    "TTCExecutor",
    "first_result_selector",
    "majority_vote_selector",
    "best_of_n_selector",
]

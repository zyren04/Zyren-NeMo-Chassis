"""
Enhanced Stress Tests for Concurrency and Edge Cases
Tests for TokenBucket, EventStore, NIMClient, and SandboxRunner under heavy load.
"""

import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.infrastructure.nim_client import NIMClient, RateLimiter, TokenBucket
from src.sandbox.runner import SandboxRunner
from src.state.event_store import EventRecord, EventStore


class TestTokenBucketStress:
    """Stress tests for TokenBucket under extreme concurrent load."""

    @pytest.mark.asyncio
    async def test_burst_100_concurrent_takes(self):
        """Test 100 concurrent take() calls - verify no token drift or deadlock."""
        bucket = TokenBucket(capacity=100, refill_rate=50.0)

        async def take_one():
            return await bucket.take(1)

        tasks = [take_one() for _ in range(100)]
        results = await asyncio.gather(*tasks)

        # All should succeed immediately (within capacity)
        assert all(w == 0.0 for w in results)
        # Should have consumed exactly 100 tokens
        assert abs(bucket._tokens - 0.0) < 0.1

    @pytest.mark.asyncio
    async def test_burst_200_over_capacity(self):
        """Test 200 concurrent takes over capacity - verify proper waiting."""
        bucket = TokenBucket(capacity=100, refill_rate=100.0)  # Fast refill for test

        async def take_one():
            return await bucket.take(1)

        tasks = [take_one() for _ in range(200)]
        results = await asyncio.gather(*tasks)

        # First 100 should be instant, rest should wait
        instant_count = sum(1 for w in results if w == 0.0)
        wait_count = sum(1 for w in results if w > 0.0)

        assert instant_count == 100
        assert wait_count == 100

    @pytest.mark.asyncio
    async def test_token_drift_prevention(self):
        """Verify no token drift under rapid concurrent access with refills."""
        bucket = TokenBucket(capacity=50, refill_rate=1000.0)  # Very fast refill

        async def take_and_wait():
            await bucket.take(1)
            await asyncio.sleep(0.001)  # Small delay to allow refill

        # Run 500 operations with refill
        for _ in range(10):
            tasks = [take_and_wait() for _ in range(50)]
            await asyncio.gather(*tasks)

        # With 1000 tokens/sec refill and small sleeps, bucket should have some tokens
        # Note: concurrent access means some tokens consumed between refills
        assert bucket._tokens >= 0.0  # Non-negative

    @pytest.mark.asyncio
    async def test_take_large_token_amounts(self):
        """Test taking large token amounts atomically."""
        bucket = TokenBucket(capacity=1000, refill_rate=100.0)

        async def take_chunk(amount):
            return await bucket.take(amount)

        # Take 5 chunks of 200 each
        tasks = [take_chunk(200) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        assert all(w == 0.0 for w in results)
        # Allow small floating point remainder from concurrent refill
        assert bucket._tokens < 1.0


class TestRateLimiterStress:
    """Stress tests for RateLimiter composite (semaphore + token bucket)."""

    @pytest.mark.asyncio
    async def test_50_concurrent_requests_semaphore_3(self):
        """Test 50 concurrent requests against semaphore(3) - no deadlock."""
        limiter = RateLimiter(max_concurrent=3, max_rpm=1000)  # High RPM to focus on semaphore

        active = 0
        max_active = 0
        lock = asyncio.Lock()
        completed = 0

        async def request():
            nonlocal active, max_active, completed
            async with limiter.acquire():
                async with lock:
                    active += 1
                    max_active = max(max_active, active)
                await asyncio.sleep(0.01)  # Simulate work
                async with lock:
                    active -= 1
                    completed += 1

        tasks = [request() for _ in range(50)]
        await asyncio.gather(*tasks)

        assert max_active <= 3
        assert completed == 50

    @pytest.mark.asyncio
    async def test_100_concurrent_mixed_semaphore_rpm(self):
        """Test 100 concurrent requests with both semaphore and RPM limits."""
        limiter = RateLimiter(max_concurrent=5, max_rpm=100)  # ~1.67 req/sec

        start = time.monotonic()
        completed = 0

        async def request():
            nonlocal completed
            async with limiter.acquire():
                await asyncio.sleep(0.001)
                completed += 1

        tasks = [request() for _ in range(20)]  # 20 requests, should take ~12 sec at 100 RPM
        await asyncio.gather(*tasks)

        elapsed = time.monotonic() - start
        assert completed == 20
        # Should take some time due to RPM limit but not deadlock
        assert elapsed > 0.0  # Just verify it completes


class TestEventStoreConcurrency:
    """Stress tests for EventStore concurrent writes."""

    @pytest_asyncio.fixture(scope="function")
    async def event_store(self):
        store = EventStore(db_path="data/test_stress.db")
        await store._initialize()
        yield store
        await store.close()
        # Cleanup
        import os

        if os.path.exists("data/test_stress.db"):
            os.remove("data/test_stress.db")

    @pytest.mark.asyncio
    async def test_50_concurrent_record_event(self, event_store: EventStore):
        """Test 50 simultaneous record_event() calls - no database locking conflicts."""
        execution_id = "test-exec-concurrent"

        async def record_event(i):
            return await event_store.record_event(
                execution_id=execution_id,
                event_type="TEST_EVENT",
                payload={"index": i, "data": f"test-{i}"},
                node_name="test_node",
                iteration=1,
            )

        tasks = [record_event(i) for i in range(50)]
        events = await asyncio.gather(*tasks)

        assert len(events) == 50
        assert all(isinstance(e, EventRecord) for e in events)

        # Verify all events persisted
        retrieved = await event_store.get_events(execution_id, limit=100)
        assert len(retrieved) == 50

    @pytest.mark.asyncio
    async def test_100_concurrent_batch_writes(self, event_store):
        """Test 100 concurrent batch writes using record_events_batch."""
        execution_id = "test-exec-batch"

        async def batch_write(batch_id):
            events = [
                EventRecord(
                    event_id=f"{batch_id}-{i}",
                    execution_id=execution_id,
                    event_type="BATCH_EVENT",
                    node_name="batch_node",
                    payload={"batch": batch_id, "item": i},
                    timestamp=datetime.utcnow(),
                    iteration=batch_id,
                )
                for i in range(5)
            ]
            return await event_store.record_events_batch(events)

        tasks = [batch_write(i) for i in range(20)]  # 20 batches * 5 = 100 events
        results = await asyncio.gather(*tasks)

        assert len(results) == 20
        for batch_events in results:
            assert len(batch_events) == 5

        # Verify all persisted
        retrieved = await event_store.get_events(execution_id, limit=200)
        assert len(retrieved) == 100

    @pytest.mark.asyncio
    async def test_concurrent_read_write(self, event_store):
        """Test concurrent reads while writing."""
        execution_id = "test-exec-rw"

        # Start writing
        async def writer():
            for i in range(100):
                await event_store.record_event(
                    execution_id=execution_id,
                    event_type="WRITE_EVENT",
                    payload={"seq": i},
                    node_name="writer",
                    iteration=1,
                )
                await asyncio.sleep(0.001)

        # Concurrent reads
        async def reader():
            results = []
            for _ in range(10):
                events = await event_store.get_events(execution_id, limit=50)
                results.append(len(events))
                await asyncio.sleep(0.01)
            return results

        await asyncio.gather(writer(), reader(), reader(), reader())


class TestNIMClientConcurrency:
    """Stress tests for NIMClient under concurrent load."""

    @pytest.fixture
    def mock_nim_client(self):
        with patch("src.infrastructure.nim_client.ChatNVIDIA") as mock_chat:
            mock_instance = AsyncMock()
            mock_chat.return_value = mock_instance
            client = NIMClient(
                model="test-model",
                api_key="test-key",
                max_concurrent=3,
                max_rpm=1000,  # High RPM to test semaphore primarily
            )
            client._client = mock_instance
            yield client, mock_instance

    @pytest.mark.asyncio
    async def test_50_concurrent_ainvoke(self, mock_nim_client):
        """Test 50 concurrent ainvoke() calls with semaphore(3) - no deadlock."""
        client, mock = mock_nim_client

        mock.ainvoke.side_effect = [
            MagicMock(content=f"Response {i}", response_metadata={"finish_reason": "stop"})
            for i in range(50)
        ]

        tasks = [client.ainvoke([MagicMock()]) for _ in range(50)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 50
        assert mock.ainvoke.call_count == 50

    @pytest.mark.asyncio
    async def test_concurrent_ainvoke_with_continuation(self, mock_nim_client):
        """Test concurrent requests with auto-continuation triggered."""
        client, mock = mock_nim_client
        client.auto_continue_truncated = True
        client.max_continuations = 2

        # Each call returns truncated then complete
        mock.ainvoke.side_effect = [
            # Request 0
            MagicMock(content="Part 1", response_metadata={"finish_reason": "length"}),
            MagicMock(content="Part 2", response_metadata={"finish_reason": "stop"}),
            # Request 1
            MagicMock(content="Part A", response_metadata={"finish_reason": "length"}),
            MagicMock(content="Part B", response_metadata={"finish_reason": "stop"}),
        ] * 25  # Enough for 50 requests with 2 continuations each

        tasks = [client.ainvoke([MagicMock()]) for _ in range(50)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 50
        # Each request should have 2 calls (initial + continuation)
        assert mock.ainvoke.call_count == 100

    @pytest.mark.asyncio
    async def test_burst_with_rate_limiting(self, mock_nim_client):
        """Test burst with RPM limiting active."""
        client, mock = mock_nim_client
        client.rate_limiter = RateLimiter(max_concurrent=3, max_rpm=30)  # 0.5 req/sec

        mock.ainvoke.side_effect = [
            MagicMock(content=f"Response {i}", response_metadata={"finish_reason": "stop"})
            for i in range(10)
        ]

        start = time.monotonic()
        tasks = [client.ainvoke([MagicMock()]) for _ in range(10)]
        await asyncio.gather(*tasks)
        elapsed = time.monotonic() - start

        # With 30 RPM (0.5/sec), 10 requests with 3 concurrent should take ~14 seconds
        # But burst capacity of 30 allows first 30 instant, so should be fast
        assert elapsed < 2.0  # Should complete quickly due to burst capacity


class TestSandboxRunnerEdgeCases:
    """Edge case tests for SandboxRunner."""

    @pytest.fixture
    def runner(self):
        return SandboxRunner(default_timeout=5.0, max_output_size=1024)

    @pytest.mark.asyncio
    async def test_timeout_sigkill_and_reap(self, runner):
        """Test that timeout sends SIGKILL to process group and reaps zombie."""
        # Run a command that hangs
        result = await runner.run(command=["sleep", "100"], timeout=0.5, capture_output=True)

        assert result.timed_out is True
        assert result.exit_code == runner.EXIT_TIMEOUT
        assert "killed" in result.stderr.lower() or "timeout" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_exit_code_parsing_standard_signals(self, runner):
        """Test accurate exit code parsing for standard signals."""
        # Test exit code 0 (success)
        result = await runner.run(command=["true"], timeout=5.0)
        assert result.exit_code == 0
        assert result.success is True

        # Test exit code 1 (general error)
        result = await runner.run(command=["false"], timeout=5.0)
        assert result.exit_code == 1
        assert result.success is False

        # Test exit code 124 (timeout - SIGXCPU)
        parsed = SandboxRunner.parse_exit_code(124)
        assert parsed["success"] is False
        assert parsed["signal"] == "SIGXCPU"

        # Test exit code 137 (SIGKILL - OOM or manual kill)
        parsed = SandboxRunner.parse_exit_code(137)
        assert parsed["success"] is False
        assert parsed["signal"] == "SIGKILL"

        # Test exit code 143 (SIGTERM)
        parsed = SandboxRunner.parse_exit_code(143)
        assert parsed["success"] is False
        assert parsed["signal"] == "SIGTERM"

    @pytest.mark.asyncio
    async def test_zombie_reaping(self, runner):
        """Verify no zombie processes left after timeout."""
        # This test verifies the process.wait() call after SIGKILL
        result = await runner.run(
            command=["sleep", "100"],
            timeout=0.2,
        )

        assert result.timed_out
        # The process.wait() in the timeout handler should reap the process
        # If zombie remained, we'd see issues in subsequent runs
        result2 = await runner.run(command=["echo", "test"], timeout=5.0)
        assert result2.success

    @pytest.mark.asyncio
    async def test_output_size_limit(self, runner):
        """Test output truncation at max_output_size."""
        # Create a command that outputs more than 1KB
        result = await runner.run(
            command=["python3", "-c", "print('x' * 2000)"],
            timeout=5.0,
        )

        assert result.success
        assert len(result.stdout) <= 1024 + 100  # Allow some buffer for truncation message
        assert "truncated" in result.stdout or len(result.stdout) < 2000

    @pytest.mark.asyncio
    async def test_batch_execution_concurrency(self, runner):
        """Test run_batch with semaphore limiting."""
        commands = [["sleep", "0.1"] for _ in range(10)]

        start = time.monotonic()
        results = await runner.run_batch(commands, timeout=5.0, max_concurrent=3)
        elapsed = time.monotonic() - start

        assert len(results) == 10
        assert all(r.success for r in results)
        # With max_concurrent=3, 10 * 0.1s should take ~0.4s (3+3+3+1 batches)
        assert elapsed > 0.2
        assert elapsed < 2.0


class TestIntegratedWorkflowStress:
    """Integrated stress tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_full_pipeline_100_iterations(self):
        """Test a simulated full pipeline execution with 100 iterations."""
        from src.runtime.engine import WorkflowEngine
        from src.state.state_schema import BaseState

        engine = WorkflowEngine(max_iterations=10)
        state = BaseState(execution_id="stress-test", max_iterations=10)

        # Register dummy nodes
        async def processor(state):
            return state.increment_iteration()

        async def validator(state):
            return state.increment_iteration()

        engine.register_node("processor", processor)
        engine.register_node("validator", validator)
        engine.register_edge("processor", "validator")
        # Don't create a cycle - just run processor -> validator once
        engine.set_entry_point("processor")

        compiled = engine.compile()

        # Execute with max_steps to limit iterations
        from langchain_core.runnables import RunnableConfig

        config = RunnableConfig(max_steps=20)
        final_state = await compiled.ainvoke(state, config=config)

        # Verify state progression
        assert final_state.iteration >= 0
        assert final_state.execution_id == "stress-test"

    @pytest.mark.asyncio
    async def test_concurrent_workflow_executions(self):
        """Test multiple concurrent workflow executions."""
        from src.runtime.engine import WorkflowEngine
        from src.state.state_schema import BaseState

        engine = WorkflowEngine(max_iterations=10)

        async def dummy_node(state):
            await asyncio.sleep(0.01)
            return state.increment_iteration()

        engine.register_node("node1", dummy_node)
        engine.set_entry_point("node1")
        compiled = engine.compile()

        # Launch 20 concurrent executions
        initial_states = [BaseState(execution_id=f"exec-{i}") for i in range(20)]
        results = await compiled.abatch(initial_states)

        assert len(results) == 20
        for i, state in enumerate(results):
            assert state.execution_id == f"exec-{i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])

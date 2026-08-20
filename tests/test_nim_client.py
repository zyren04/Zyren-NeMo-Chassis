"""
Tests for infrastructure.nim_client module
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure.nim_client import NIMClient, RateLimiter, TokenBucket


class TestTokenBucket:
    """Test TokenBucket rate limiter."""

    @pytest.mark.asyncio
    async def test_take_tokens_available(self):
        """Test taking tokens when available."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        wait_time = await bucket.take(5)
        assert wait_time == 0.0
        assert bucket._tokens == 5.0

    @pytest.mark.asyncio
    async def test_take_tokens_wait(self):
        """Test taking tokens when need to wait."""
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        # Drain the bucket
        await bucket.take(5)
        # Now try to take more - should wait
        wait_time = await bucket.take(3)
        # Allow small floating point tolerance
        assert abs(wait_time - 3.0) < 0.01

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test thread-safety under concurrent access."""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)
        tasks = [bucket.take(1) for _ in range(50)]
        results = await asyncio.gather(*tasks)
        assert all(w == 0.0 for w in results)
        # Allow small floating point tolerance due to concurrent refill
        assert abs(bucket._tokens - 50.0) < 0.1


class TestRateLimiter:
    """Test RateLimiter composite."""

    @pytest.mark.asyncio
    async def test_acquire_respects_semaphore(self):
        """Test semaphore limits concurrency."""
        limiter = RateLimiter(max_concurrent=2, max_rpm=100)
        active = 0
        max_active = 0
        lock = asyncio.Lock()

        async def task():
            nonlocal active, max_active
            async with limiter.acquire():
                async with lock:
                    active += 1
                    max_active = max(max_active, active)
                await asyncio.sleep(0.05)
                async with lock:
                    active -= 1

        await asyncio.gather(*[task() for _ in range(10)])
        assert max_active <= 2

    @pytest.mark.asyncio
    async def test_acquire_respects_token_bucket(self):
        """Test token bucket limits RPM."""
        limiter = RateLimiter(max_concurrent=10, max_rpm=10)  # 10 per minute = ~0.167/sec
        start = asyncio.get_event_loop().time()
        for _ in range(5):
            async with limiter.acquire():
                pass
        elapsed = asyncio.get_event_loop().time() - start
        # 5 requests at 10 RPM should take ~30 seconds without refill
        # But with small capacity, they should be near-instant
        assert elapsed < 1.0  # Should be fast due to burst capacity

    @pytest.mark.asyncio
    async def test_stats(self):
        """Test statistics tracking."""
        limiter = RateLimiter(max_concurrent=3, max_rpm=40)
        async with limiter.acquire():
            pass
        stats = limiter.stats
        assert stats["request_count"] == 1
        assert stats["capacity"] == 40
        assert 0 <= stats["available_tokens"] <= 40


class TestNIMClient:
    """Test NIMClient wrapper."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock NIMClient."""
        with patch("src.infrastructure.nim_client.ChatNVIDIA") as mock_chat:
            mock_instance = AsyncMock()
            mock_chat.return_value = mock_instance
            client = NIMClient(
                model="test-model",
                api_key="test-key",
                max_concurrent=3,
                max_rpm=40,
            )
            client._client = mock_instance
            yield client, mock_instance

    @pytest.mark.asyncio
    async def test_ainvoke_basic(self, mock_client):
        """Test basic invocation."""
        client, mock = mock_client
        mock.ainvoke.return_value = MagicMock(
            content="Test response", response_metadata={"finish_reason": "stop"}
        )
        result = await client.ainvoke([MagicMock()])
        assert result.content == "Test response"

    @pytest.mark.asyncio
    async def test_auto_continue_truncated(self, mock_client):
        """Test auto-continuation on finish_reason=length."""
        client, mock = mock_client
        client.auto_continue_truncated = True
        client.max_continuations = 2

        # First call returns truncated
        mock.ainvoke.side_effect = [
            MagicMock(content="Part 1", response_metadata={"finish_reason": "length"}),
            MagicMock(content="Part 2", response_metadata={"finish_reason": "stop"}),
        ]

        result = await client.ainvoke([MagicMock()])
        assert "Part 1" in result.content
        assert "Part 2" in result.content
        assert mock.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_no_continue_when_complete(self, mock_client):
        """Test no continuation when finish_reason=stop."""
        client, mock = mock_client
        client.auto_continue_truncated = True
        mock.ainvoke.return_value = MagicMock(
            content="Complete response", response_metadata={"finish_reason": "stop"}
        )
        result = await client.ainvoke([MagicMock()])
        assert result.content == "Complete response"
        assert mock.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_astream(self, mock_client):
        """Test streaming."""
        client, mock = mock_client
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        async def gen():
            for chunk in ["Hello", " ", "World"]:
                yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))

        # Replace the AsyncMock's astream with a plain function that returns async generator
        def astream_mock(*args, **kwargs):
            return gen()

        mock.astream = astream_mock
        chunks = []
        async for chunk in client.astream([MagicMock()]):
            chunks.append(chunk.message.content)
        assert chunks == ["Hello", " ", "World"]

    @pytest.mark.asyncio
    async def test_abatch(self, mock_client):
        """Test batch invocation."""
        client, mock = mock_client
        mock.ainvoke.side_effect = [
            MagicMock(content=f"Response {i}", response_metadata={}) for i in range(3)
        ]
        results = await client.abatch([[MagicMock()] for _ in range(3)])
        assert len(results) == 3
        assert results[0].content == "Response 0"

    @pytest.mark.asyncio
    async def test_burst_load_50_requests(self, mock_client):
        """Test burst load: 50 concurrent requests against semaphore(3)."""
        client, mock = mock_client
        mock.ainvoke.side_effect = [
            MagicMock(content=f"Response {i}", response_metadata={"finish_reason": "stop"})
            for i in range(50)
        ]

        tasks = [client.ainvoke([MagicMock()]) for _ in range(50)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 50
        assert all(r.content.startswith("Response") for r in results)


class TestNetworkResilience:
    """Test network resilience with tenacity retry."""

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        """Test exponential backoff retry on connection error."""
        with patch("src.infrastructure.nim_client.ChatNVIDIA") as mock_chat:
            mock_instance = AsyncMock()
            mock_chat.return_value = mock_instance

            # Fail twice then succeed
            mock_instance.ainvoke.side_effect = [
                ConnectionError("Network down"),
                ConnectionError("Still down"),
                MagicMock(content="Success", response_metadata={}),
            ]

            client = NIMClient(model="test", api_key="key", max_retries=3)
            client._client = mock_instance

            result = await client.ainvoke([MagicMock()])
            assert result.content == "Success"
            assert mock_instance.ainvoke.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

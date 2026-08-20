"""
Pytest Configuration and Shared Fixtures
Provides mocks for all external dependencies to enable 100% offline testing.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# =============================================================================
# Mock NVIDIA Client
# =============================================================================


class MockNVIDIAClient:
    """Mock NVIDIA NIM client for offline testing."""

    def __init__(self, **kwargs):
        self.model = kwargs.get("model", "mock-model")
        self.call_count = 0
        self.last_messages = None
        self.last_kwargs = None
        self._response_text = "Mock response from NVIDIA NIM"
        self._should_fail = False
        self._failure_exception = None

    def set_response(self, text: str):
        self._response_text = text

    def set_failure(self, exception: Exception):
        self._should_fail = True
        self._failure_exception = exception

    async def ainvoke(self, messages: list[BaseMessage], **kwargs) -> ChatResult:
        self.call_count += 1
        self.last_messages = messages
        self.last_kwargs = kwargs

        if self._should_fail:
            raise self._failure_exception

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self._response_text))]
        )

    async def abatch(self, messages_list: list[list[BaseMessage]], **kwargs) -> list[ChatResult]:
        return [await self.ainvoke(m, **kwargs) for m in messages_list]

    async def astream(self, messages: list[BaseMessage], **kwargs) -> AsyncIterator[ChatGeneration]:
        self.call_count += 1
        self.last_messages = messages
        self.last_kwargs = kwargs

        if self._should_fail:
            raise self._failure_exception

        for chunk in ["Mock ", "response ", "from ", "NVIDIA ", "NIM"]:
            yield ChatGeneration(message=AIMessage(content=chunk))
            await asyncio.sleep(0.01)

    async def ainvoke_simple(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        self.call_count += 1
        if self._should_fail:
            raise self._failure_exception
        return self._response_text

    def get_stats(self) -> dict[str, Any]:
        return {"model": self.model, "call_count": self.call_count}


@pytest.fixture
def mock_nvidia_client():
    """Provide a mock NVIDIA client."""
    return MockNVIDIAClient()


# =============================================================================
# NeMo Relay Context Isolation Fixtures
# =============================================================================


@pytest.fixture
def isolated_nemo_relay_context():
    """Provide a clean NeMo Relay context for each test."""
    import contextvars

    from src.infrastructure.nemo_relay_integration import get_nemo_relay_integration

    integration = get_nemo_relay_integration()
    # Create fresh isolated stack for this test
    integration.create_isolated_scope_stack()
    child_context = integration.fork_asyncio_context()

    # Run test in isolated context
    contextvars.copy_context().run(child_context.run, lambda: None)

    yield integration

    # Cleanup
    integration.get_scope_stack()  # Reset to default


@pytest.fixture
def nemo_relay_integration():
    """Provide NeMoRelayIntegration instance."""
    from src.infrastructure.nemo_relay_integration import get_nemo_relay_integration

    return get_nemo_relay_integration()

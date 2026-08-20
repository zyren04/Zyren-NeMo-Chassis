"""
Tests for routing module — Switchyard Router
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from src.routing.router import ModelTarget, SwitchyardRouter, get_router, set_router


class TestModelTarget:
    """Test ModelTarget dataclass validation."""

    def test_valid_target(self):
        target = ModelTarget(
            name="test",
            model="nvidia/test-model",
            max_rpm=60,
            rate_limit_mode="token_bucket",
            tags=["test"],
            enabled=True,
        )
        assert target.name == "test"
        assert target.max_rpm == 60

    def test_invalid_rate_limit_mode(self):
        with pytest.raises(ValueError, match="Invalid rate_limit_mode"):
            ModelTarget(
                name="test",
                model="nvidia/test",
                rate_limit_mode="invalid_mode",
            )

    def test_invalid_max_rpm(self):
        with pytest.raises(ValueError, match="max_rpm must be positive"):
            ModelTarget(name="test", model="nvidia/test", max_rpm=0)

    def test_invalid_negative_max_rpm(self):
        with pytest.raises(ValueError, match="max_rpm must be positive"):
            ModelTarget(name="test", model="nvidia/test", max_rpm=-10)


class TestSwitchyardRouter:
    """Test SwitchyardRouter functionality."""

    @pytest.fixture
    def sample_config(self, tmp_path):
        """Create a temporary config file for testing."""
        config = {
            "version": "1.0",
            "models": [
                {
                    "name": "default",
                    "model": "nvidia/nemotron-3-8b",
                    "max_rpm": 100,
                    "rate_limit_mode": "token_bucket",
                    "tags": ["general"],
                    "enabled": True,
                },
                {
                    "name": "reasoning",
                    "model": "nvidia/nemotron-3-ultra",
                    "max_rpm": 30,
                    "rate_limit_mode": "strict",
                    "tags": ["reasoning"],
                    "enabled": True,
                },
                {
                    "name": "coding",
                    "model": "nvidia/code-llama-70b",
                    "max_rpm": 40,
                    "rate_limit_mode": "token_bucket",
                    "tags": ["coding"],
                    "enabled": True,
                },
                {
                    "name": "disabled-model",
                    "model": "nvidia/old-model",
                    "max_rpm": 60,
                    "rate_limit_mode": "token_bucket",
                    "tags": ["deprecated"],
                    "enabled": False,
                },
            ],
        }
        config_path = tmp_path / "models.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        return str(config_path)

    @pytest.fixture
    def router(self, sample_config):
        """Create a router instance with test config."""
        with patch("src.routing.router.NIMClient") as mock_client_class:
            # Return different mock instances for each call
            mock_clients = {}
            def create_mock(*args, **kwargs):
                model = kwargs.get('model', 'unknown')
                if model not in mock_clients:
                    mock_clients[model] = MagicMock()
                return mock_clients[model]
            mock_client_class.side_effect = create_mock
            router = SwitchyardRouter(sample_config)
            yield router

    def test_load_config(self, router):
        """Test YAML loading and disabled model filtering."""
        assert "default" in router._targets
        assert "reasoning" in router._targets
        assert "coding" in router._targets
        assert "disabled-model" not in router._targets
        assert len(router._targets) == 3

    def test_get_client_creates_and_caches(self, router):
        """Test client creation and caching."""
        client1 = router.get_client("default")
        client2 = router.get_client("default")
        assert client1 is client2

    def test_get_client_different_targets(self, router):
        """Test different targets get different clients."""
        client_default = router.get_client("default")
        client_reasoning = router.get_client("reasoning")
        assert client_default is not client_reasoning

    def test_get_client_fallback_to_default(self, router):
        """Test fallback to default for unknown target."""
        client = router.get_client("unknown-target")
        default_client = router.get_client("default")
        assert client is default_client

    def test_get_client_raises_for_missing_default(self, tmp_path):
        """Test KeyError when no default and target missing."""
        config = {
            "version": "1.0",
            "models": [
                {
                    "name": "reasoning",
                    "model": "nvidia/nemotron-3-ultra",
                    "max_rpm": 30,
                    "rate_limit_mode": "strict",
                    "tags": ["reasoning"],
                    "enabled": True,
                },
            ],
        }
        config_path = tmp_path / "models.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        with patch("src.routing.router.NIMClient"):
            router = SwitchyardRouter(str(config_path))
            with pytest.raises(KeyError):
                router.get_client("unknown")

    def test_list_targets(self, router):
        """Test listing enabled targets."""
        targets = router.list_targets()
        assert set(targets) == {"default", "reasoning", "coding"}

    def test_get_target(self, router):
        """Test getting target configuration."""
        target = router.get_target("reasoning")
        assert target is not None
        assert target.name == "reasoning"
        assert target.model == "nvidia/nemotron-3-ultra"
        assert target.rate_limit_mode == "strict"

        assert router.get_target("disabled-model") is None
        assert router.get_target("nonexistent") is None

    def test_reload(self, router, sample_config, tmp_path):
        """Test configuration reload picks up changes."""
        with open(sample_config) as f:
            config = yaml.safe_load(f)
        config["models"].append(
            {
                "name": "new-model",
                "model": "nvidia/new-model",
                "max_rpm": 50,
                "rate_limit_mode": "token_bucket",
                "tags": ["new"],
                "enabled": True,
            }
        )
        with open(sample_config, "w") as f:
            yaml.safe_dump(config, f)

        with patch("src.routing.router.NIMClient"):
            router.reload()
            assert "new-model" in router._targets


class TestRoutingLogic:
    """Test autonomous routing heuristics."""

    @pytest.fixture
    def router(self, tmp_path):
        config = {
            "version": "1.0",
            "models": [
                {"name": "default", "model": "nvidia/test", "max_rpm": 100, "rate_limit_mode": "token_bucket", "tags": [], "enabled": True},
                {"name": "reasoning", "model": "nvidia/reasoning", "max_rpm": 30, "rate_limit_mode": "strict", "tags": [], "enabled": True},
                {"name": "coding", "model": "nvidia/coding", "max_rpm": 40, "rate_limit_mode": "token_bucket", "tags": [], "enabled": True},
                {"name": "fast", "model": "nvidia/fast", "max_rpm": 200, "rate_limit_mode": "token_bucket", "tags": [], "enabled": True},
            ],
        }
        config_path = tmp_path / "models.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        with patch("src.routing.router.NIMClient"):
            return SwitchyardRouter(str(config_path))

    def test_auto_route_coding_keywords(self, router):
        """Test coding keywords route to coding target."""
        assert router._auto_route("write a python function") == "coding"
        assert router._auto_route("debug this code") == "coding"
        assert router._auto_route("implement a class") == "coding"
        assert router._auto_route("SQL query for users") == "coding"
        assert router._auto_route("refactor this script") == "coding"

    def test_auto_route_reasoning_keywords(self, router):
        """Test reasoning keywords route to reasoning target."""
        assert router._auto_route("analyze the tradeoffs") == "reasoning"
        assert router._auto_route("reason step by step") == "reasoning"
        assert router._auto_route("prove this theorem") == "reasoning"
        assert router._auto_route("explain why this works") == "reasoning"
        assert router._auto_route("design an architecture") == "reasoning"

    def test_auto_route_fast_keywords(self, router):
        """Test fast/summarize keywords route to fast target."""
        assert router._auto_route("summarize this text") == "fast"
        assert router._auto_route("brief summary") == "fast"
        assert router._auto_route("tldr") == "fast"
        assert router._auto_route("one sentence answer") == "fast"

    def test_auto_route_default_fallback(self, router):
        """Test unknown prompts fall back to default."""
        assert router._auto_route("hello world") == "default"
        assert router._auto_route("what is the weather") == "default"

    def test_explicit_task_type(self, router):
        """Test explicit task_type bypasses auto-routing."""
        assert router._resolve_target("reasoning") == "reasoning"
        assert router._resolve_target("coding") == "coding"
        assert router._resolve_target("fast") == "fast"
        assert router._resolve_target("custom-target") == "custom-target"


class TestRouteMethod:
    """Test the async route() method with mocked NIMClient."""

    @pytest.fixture
    def router_with_mock(self, tmp_path):
        config = {
            "version": "1.0",
            "models": [
                {"name": "default", "model": "nvidia/test", "max_rpm": 100, "rate_limit_mode": "token_bucket", "tags": [], "enabled": True},
                {"name": "reasoning", "model": "nvidia/reasoning", "max_rpm": 30, "rate_limit_mode": "strict", "tags": [], "enabled": True},
            ],
        }
        config_path = tmp_path / "models.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        with patch("src.routing.router.NIMClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.ainvoke_simple = AsyncMock(return_value="Mocked response")
            mock_client_class.return_value = mock_client
            router = SwitchyardRouter(str(config_path))
            yield router, mock_client

    @pytest.mark.asyncio
    async def test_route_auto(self, router_with_mock):
        """Test route() with auto task_type."""
        router, mock_client = router_with_mock
        result = await router.route("write a python function", task_type="auto")
        assert result == "Mocked response"
        mock_client.ainvoke_simple.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_explicit(self, router_with_mock):
        """Test route() with explicit task_type."""
        router, mock_client = router_with_mock
        result = await router.route("analyze this", task_type="reasoning")
        assert result == "Mocked response"

    @pytest.mark.asyncio
    async def test_route_passes_system_prompt(self, router_with_mock):
        """Test route() passes system_prompt to client."""
        router, mock_client = router_with_mock
        await router.route("test", system_prompt="You are helpful")
        mock_client.ainvoke_simple.assert_called_with("test", system_prompt="You are helpful")

    @pytest.mark.asyncio
    async def test_route_passes_kwargs(self, router_with_mock):
        """Test route() passes extra kwargs to client."""
        router, mock_client = router_with_mock
        await router.route("test", temperature=0.7, max_tokens=100)
        mock_client.ainvoke_simple.assert_called_with("test", system_prompt=None, temperature=0.7, max_tokens=100)


class TestGlobalRouter:
    """Test global router singleton."""

    def test_get_router_singleton(self, tmp_path):
        """Test get_router returns same instance."""
        config = {
            "version": "1.0",
            "models": [
                {"name": "default", "model": "nvidia/test", "max_rpm": 100, "rate_limit_mode": "token_bucket", "tags": [], "enabled": True},
            ],
        }
        config_path = tmp_path / "models.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        with patch("src.routing.router.NIMClient"):
            router1 = get_router(str(config_path))
            router2 = get_router(str(config_path))
            assert router1 is router2

    def test_set_router(self, tmp_path):
        """Test set_router for dependency injection."""
        config = {
            "version": "1.0",
            "models": [
                {"name": "default", "model": "nvidia/test", "max_rpm": 100, "rate_limit_mode": "token_bucket", "tags": [], "enabled": True},
            ],
        }
        config_path = tmp_path / "models.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        with patch("src.routing.router.NIMClient"):
            get_router(str(config_path))  # Initialize singleton
            router2 = SwitchyardRouter(str(config_path))
            set_router(router2)
            assert get_router() is router2


class TestConfigErrors:
    """Test configuration error handling."""

    def test_missing_config_file(self):
        """Test FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError):
            SwitchyardRouter("nonexistent.yaml")

    def test_missing_models_key(self, tmp_path):
        """Test ValueError for missing models key."""
        config_path = tmp_path / "models.yaml"
        config_path.write_text("version: '1.0'\nother_key: []\n")

        with pytest.raises(ValueError, match="missing 'models' key"):
            SwitchyardRouter(str(config_path))

    def test_no_enabled_models(self, tmp_path):
        """Test ValueError when no models enabled."""
        config = {
            "version": "1.0",
            "models": [
                {"name": "disabled", "model": "nvidia/test", "max_rpm": 100, "rate_limit_mode": "token_bucket", "tags": [], "enabled": False},
            ],
        }
        config_path = tmp_path / "models.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        with pytest.raises(ValueError, match="No enabled model targets"):
            SwitchyardRouter(str(config_path))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


"""Tests for the config module."""

import os
import pytest
from unittest.mock import patch

from ollama_agent.config import Config, _parse_bool


class TestParseBool:
    """Tests for _parse_bool function."""

    def test_parse_true_values(self):
        assert _parse_bool("true") is True
        assert _parse_bool("True") is True
        assert _parse_bool("TRUE") is True
        assert _parse_bool("1") is True
        assert _parse_bool("yes") is True
        assert _parse_bool("Yes") is True

    def test_parse_false_values(self):
        assert _parse_bool("false") is False
        assert _parse_bool("False") is False
        assert _parse_bool("0") is False
        assert _parse_bool("no") is False
        assert _parse_bool("random") is False

    def test_parse_none_with_default(self):
        assert _parse_bool(None, default=True) is True
        assert _parse_bool(None, default=False) is False

    def test_parse_empty_string(self):
        assert _parse_bool("", default=True) is True
        assert _parse_bool("", default=False) is False


class TestConfig:
    """Tests for Config class."""

    def test_default_values(self):
        """Test that Config has sensible defaults."""
        config = Config()
        assert config.ollama_model == os.getenv("OLLAMA_MODEL", "llama3.2")
        assert config.temperature == float(os.getenv("TEMPERATURE", "0.7"))
        assert config.max_iterations == int(os.getenv("MAX_ITERATIONS", "10"))
        assert config.max_search_results == int(os.getenv("MAX_SEARCH_RESULTS", "5"))

    def test_blocked_commands_default(self):
        """Test that blocked commands list is populated."""
        config = Config()
        assert isinstance(config.blocked_commands, list)
        assert len(config.blocked_commands) > 0
        assert "rm -rf /" in config.blocked_commands

    @patch.dict(os.environ, {"OLLAMA_MODEL": "test-model"})
    def test_env_override_model(self):
        """Test that environment variables override defaults."""
        config = Config()
        assert config.ollama_model == "test-model"

    @patch.dict(os.environ, {"TEMPERATURE": "0.5"})
    def test_env_override_temperature(self):
        """Test temperature env override."""
        config = Config()
        assert config.temperature == 0.5

    @patch.dict(os.environ, {"REQUIRE_APPROVAL_COMMANDS": "false"})
    def test_env_override_approval(self):
        """Test approval settings env override."""
        config = Config()
        assert config.require_approval_commands is False

"""Tests for the config module."""

import pytest

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
        # Empty string is not in ("true", "1", "yes") so returns False
        assert _parse_bool("") is False
        assert _parse_bool("", default=True) is False  # default only used for None


class TestConfig:
    """Tests for Config class."""

    def test_default_values(self):
        """Test that Config has sensible defaults."""
        config = Config()
        # These are the defaults when no env vars are set
        assert isinstance(config.ollama_model, str)
        assert isinstance(config.temperature, float)
        assert isinstance(config.max_iterations, int)
        assert isinstance(config.max_search_results, int)

    def test_blocked_commands_default(self):
        """Test that blocked commands list is populated."""
        config = Config()
        assert isinstance(config.blocked_commands, list)
        assert len(config.blocked_commands) > 0
        assert "rm -rf /" in config.blocked_commands

    def test_config_attributes_exist(self):
        """Test all expected attributes exist."""
        config = Config()
        assert hasattr(config, "ollama_model")
        assert hasattr(config, "ollama_base_url")
        assert hasattr(config, "temperature")
        assert hasattr(config, "max_iterations")
        assert hasattr(config, "max_search_results")
        assert hasattr(config, "require_approval_commands")
        assert hasattr(config, "require_approval_files")
        assert hasattr(config, "blocked_commands")

    def test_config_is_dataclass(self):
        """Test Config is a dataclass."""
        from dataclasses import is_dataclass
        assert is_dataclass(Config)

    def test_base_url_is_string(self):
        """Test base_url is a valid string."""
        config = Config()
        assert isinstance(config.ollama_base_url, str)
        assert config.ollama_base_url.startswith("http")

    def test_approval_settings_are_bool(self):
        """Test approval settings are booleans."""
        config = Config()
        assert isinstance(config.require_approval_commands, bool)
        assert isinstance(config.require_approval_files, bool)

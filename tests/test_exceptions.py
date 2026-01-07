"""Tests for the exceptions module."""

import pytest

from ollama_agent.exceptions import (
    OllamaAgentError,
    ToolNotFoundError,
    ToolExecutionError,
    ToolRegistrationError,
    ConfigurationError,
    ApprovalDeniedError,
)


class TestOllamaAgentError:
    """Tests for base exception."""

    def test_base_exception(self):
        with pytest.raises(OllamaAgentError):
            raise OllamaAgentError("Test error")

    def test_inheritance(self):
        assert issubclass(OllamaAgentError, Exception)


class TestToolNotFoundError:
    """Tests for ToolNotFoundError."""

    def test_error_message(self):
        error = ToolNotFoundError("my_tool")
        assert "my_tool" in str(error)
        assert error.tool_name == "my_tool"

    def test_inheritance(self):
        assert issubclass(ToolNotFoundError, OllamaAgentError)

    def test_raises_correctly(self):
        with pytest.raises(ToolNotFoundError) as exc_info:
            raise ToolNotFoundError("missing_tool")
        assert exc_info.value.tool_name == "missing_tool"


class TestToolExecutionError:
    """Tests for ToolExecutionError."""

    def test_error_message(self):
        error = ToolExecutionError("my_tool", "Something went wrong")
        assert "my_tool" in str(error)
        assert "Something went wrong" in str(error)
        assert error.tool_name == "my_tool"

    def test_inheritance(self):
        assert issubclass(ToolExecutionError, OllamaAgentError)


class TestToolRegistrationError:
    """Tests for ToolRegistrationError."""

    def test_error_message(self):
        error = ToolRegistrationError("my_tool", "Already exists")
        assert "my_tool" in str(error)
        assert "Already exists" in str(error)
        assert error.tool_name == "my_tool"

    def test_inheritance(self):
        assert issubclass(ToolRegistrationError, OllamaAgentError)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_error_message(self):
        error = ConfigurationError("Invalid config")
        assert "Invalid config" in str(error)

    def test_inheritance(self):
        assert issubclass(ConfigurationError, OllamaAgentError)


class TestApprovalDeniedError:
    """Tests for ApprovalDeniedError."""

    def test_error_message(self):
        error = ApprovalDeniedError("run_command")
        assert "run_command" in str(error)
        assert error.tool_name == "run_command"

    def test_inheritance(self):
        assert issubclass(ApprovalDeniedError, OllamaAgentError)

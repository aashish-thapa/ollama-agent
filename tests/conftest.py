"""Pytest configuration and fixtures."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    mock_response = MagicMock()
    mock_response.content = "This is a test response."
    return mock_response


@pytest.fixture
def mock_tool_response():
    """Create a mock LLM response with tool call."""
    mock_response = MagicMock()
    mock_response.content = "TOOL: get_current_time\nINPUT:"
    return mock_response


@pytest.fixture
def mock_chat_ollama():
    """Mock ChatOllama to avoid actual API calls."""
    with patch("ollama_agent.agent.ChatOllama") as mock:
        yield mock


@pytest.fixture
def sample_tools():
    """Sample tools dictionary for testing."""
    return {
        "test_tool": {
            "func": lambda x: f"Result: {x}",
            "description": "A test tool",
        },
        "no_input_tool": {
            "func": lambda: "No input result",
            "description": "A tool that takes no input",
        },
    }

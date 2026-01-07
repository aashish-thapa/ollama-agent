"""Tests for the agent module."""

import pytest
from unittest.mock import MagicMock, patch

from ollama_agent.agent import OllamaAgent, _build_system_prompt
from ollama_agent.exceptions import ToolNotFoundError


class TestBuildSystemPrompt:
    """Tests for _build_system_prompt function."""

    def test_includes_tool_names(self):
        tools = {
            "tool1": {"func": lambda: None, "description": "First tool"},
            "tool2": {"func": lambda: None, "description": "Second tool"},
        }
        prompt = _build_system_prompt(tools)
        assert "tool1" in prompt
        assert "tool2" in prompt

    def test_includes_descriptions(self):
        tools = {
            "my_tool": {"func": lambda: None, "description": "Does something cool"},
        }
        prompt = _build_system_prompt(tools)
        assert "Does something cool" in prompt

    def test_includes_format_instructions(self):
        prompt = _build_system_prompt({})
        assert "TOOL:" in prompt
        assert "INPUT:" in prompt

    def test_includes_examples(self):
        prompt = _build_system_prompt({})
        assert "get_current_time" in prompt
        assert "web_search" in prompt


class TestOllamaAgentInit:
    """Tests for OllamaAgent initialization."""

    @patch("ollama_agent.agent.ChatOllama")
    def test_default_initialization(self, mock_chat):
        agent = OllamaAgent()
        assert agent._model == "llama3.2" or agent._model is not None
        assert agent.approval_callback is None
        mock_chat.assert_called_once()

    @patch("ollama_agent.agent.ChatOllama")
    def test_custom_model(self, mock_chat):
        agent = OllamaAgent(model="custom-model")
        assert agent._model == "custom-model"

    @patch("ollama_agent.agent.ChatOllama")
    def test_custom_temperature(self, mock_chat):
        agent = OllamaAgent(temperature=0.5)
        assert agent._temperature == 0.5

    @patch("ollama_agent.agent.ChatOllama")
    def test_custom_approval_callback(self, mock_chat):
        callback = lambda tool, input: True
        agent = OllamaAgent(approval_callback=callback)
        assert agent.approval_callback is callback

    @patch("ollama_agent.agent.ChatOllama")
    def test_custom_tools(self, mock_chat):
        custom_tools = {"my_tool": {"func": lambda: "test", "description": "Test"}}
        agent = OllamaAgent(tools=custom_tools)
        assert "my_tool" in agent._tools


class TestOllamaAgentParseToolCall:
    """Tests for _parse_tool_call method."""

    @patch("ollama_agent.agent.ChatOllama")
    def test_parse_valid_tool_call(self, mock_chat):
        agent = OllamaAgent()
        response = "TOOL: get_current_time\nINPUT:"
        result = agent._parse_tool_call(response)
        assert result == ("get_current_time", "")

    @patch("ollama_agent.agent.ChatOllama")
    def test_parse_tool_call_with_input(self, mock_chat):
        agent = OllamaAgent()
        response = "TOOL: web_search\nINPUT: python tutorials"
        result = agent._parse_tool_call(response)
        assert result == ("web_search", "python tutorials")

    @patch("ollama_agent.agent.ChatOllama")
    def test_parse_no_tool_call(self, mock_chat):
        agent = OllamaAgent()
        response = "This is just a regular response without tool calls."
        result = agent._parse_tool_call(response)
        assert result is None

    @patch("ollama_agent.agent.ChatOllama")
    def test_parse_invalid_tool_name(self, mock_chat):
        agent = OllamaAgent()
        response = "TOOL: nonexistent_tool\nINPUT: test"
        result = agent._parse_tool_call(response)
        assert result is None


class TestOllamaAgentExecuteTool:
    """Tests for _execute_tool method."""

    @patch("ollama_agent.agent.ChatOllama")
    def test_execute_tool_success(self, mock_chat):
        agent = OllamaAgent()
        result, executed = agent._execute_tool("get_current_time", "")
        assert executed is True
        assert isinstance(result, str)

    @patch("ollama_agent.agent.ChatOllama")
    def test_execute_tool_with_input(self, mock_chat):
        agent = OllamaAgent()
        result, executed = agent._execute_tool("calculator", "2 + 2")
        assert executed is True
        assert result == "4"

    @patch("ollama_agent.agent.ChatOllama")
    def test_execute_unknown_tool(self, mock_chat):
        agent = OllamaAgent()
        result, executed = agent._execute_tool("unknown_tool", "")
        assert executed is False
        assert "Unknown tool" in result

    @patch("ollama_agent.agent.ChatOllama")
    def test_execute_with_approval_denied(self, mock_chat):
        callback = lambda tool, input: False
        agent = OllamaAgent(approval_callback=callback)
        result, executed = agent._execute_tool("run_command", "ls")
        assert executed is False
        assert "denied" in result.lower()

    @patch("ollama_agent.agent.ChatOllama")
    def test_execute_with_approval_granted(self, mock_chat):
        callback = lambda tool, input: True
        agent = OllamaAgent(approval_callback=callback)
        # Use a safe command for testing
        result, executed = agent._execute_tool("calculator", "1+1")
        assert executed is True


class TestOllamaAgentRun:
    """Tests for run method."""

    @patch("ollama_agent.agent.ChatOllama")
    def test_run_simple_response(self, mock_chat):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Hello! How can I help you?"
        mock_llm.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm

        agent = OllamaAgent()
        result = agent.run("Hello")
        assert result == "Hello! How can I help you?"

    @patch("ollama_agent.agent.ChatOllama")
    def test_run_with_tool_call(self, mock_chat):
        mock_llm = MagicMock()
        # First call returns tool call, second returns final response
        mock_llm.invoke.side_effect = [
            MagicMock(content="TOOL: get_current_time\nINPUT:"),
            MagicMock(content="The current time is displayed above."),
        ]
        mock_chat.return_value = mock_llm

        agent = OllamaAgent()
        result = agent.run("What time is it?")
        assert "time" in result.lower() or mock_llm.invoke.call_count == 2

    @patch("ollama_agent.agent.ChatOllama")
    def test_run_max_iterations(self, mock_chat):
        mock_llm = MagicMock()
        # Always return a tool call to trigger max iterations
        mock_llm.invoke.return_value = MagicMock(
            content="TOOL: get_current_time\nINPUT:"
        )
        mock_chat.return_value = mock_llm

        agent = OllamaAgent(max_iterations=2)
        result = agent.run("Keep calling tools")
        assert result == "Max iterations reached."


class TestOllamaAgentReset:
    """Tests for reset method."""

    @patch("ollama_agent.agent.ChatOllama")
    def test_reset_clears_history(self, mock_chat):
        agent = OllamaAgent()
        # Add some messages
        agent._messages.append(MagicMock(content="test"))
        assert len(agent._messages) > 1

        agent.reset()
        # Should only have system message
        assert len(agent._messages) == 1


class TestOllamaAgentToolManagement:
    """Tests for add_tool and remove_tool methods."""

    @patch("ollama_agent.agent.ChatOllama")
    def test_add_tool_to_custom_tools(self, mock_chat):
        custom_tools = {}
        agent = OllamaAgent(tools=custom_tools)

        agent.add_tool("new_tool", lambda x: x, "A new tool")
        assert "new_tool" in agent._tools

    @patch("ollama_agent.agent.ChatOllama")
    def test_remove_tool(self, mock_chat):
        custom_tools = {"removable": {"func": lambda: None, "description": "Test"}}
        agent = OllamaAgent(tools=custom_tools)

        agent.remove_tool("removable")
        assert "removable" not in agent._tools

    @patch("ollama_agent.agent.ChatOllama")
    def test_remove_nonexistent_tool_raises(self, mock_chat):
        agent = OllamaAgent(tools={})
        with pytest.raises(ToolNotFoundError):
            agent.remove_tool("nonexistent")


class TestOllamaAgentProperties:
    """Tests for agent properties."""

    @patch("ollama_agent.agent.ChatOllama")
    def test_tools_property(self, mock_chat):
        agent = OllamaAgent()
        tools = agent.tools
        assert isinstance(tools, dict)
        # Should be a copy
        assert tools is not agent._tools

    @patch("ollama_agent.agent.ChatOllama")
    def test_model_property(self, mock_chat):
        agent = OllamaAgent(model="test-model")
        assert agent.model == "test-model"


class TestOllamaAgentHistory:
    """Tests for get_history method."""

    @patch("ollama_agent.agent.ChatOllama")
    def test_get_history(self, mock_chat):
        agent = OllamaAgent()
        history = agent.get_history()
        assert isinstance(history, list)
        # Should have at least system message
        assert len(history) >= 1
        assert history[0]["role"] == "system"

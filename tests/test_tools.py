"""Tests for the tools module."""

import pytest
from unittest.mock import patch, MagicMock

from ollama_agent.tools import (
    TOOLS,
    get_approval_type,
    is_command_blocked,
    register_tool,
    register_tool_func,
    unregister_tool,
    get_tool,
    get_all_tools,
    list_tools,
    _TOOLS,
)
from ollama_agent.exceptions import ToolNotFoundError, ToolRegistrationError


class TestGetApprovalType:
    """Tests for get_approval_type function."""

    def test_run_command_needs_approval(self):
        assert get_approval_type("run_command") == "commands"

    def test_write_file_needs_approval(self):
        assert get_approval_type("write_file") == "files"

    def test_safe_tool_no_approval(self):
        assert get_approval_type("get_current_time") is None
        assert get_approval_type("calculator") is None

    def test_unknown_tool_no_approval(self):
        assert get_approval_type("nonexistent_tool") is None


class TestIsCommandBlocked:
    """Tests for is_command_blocked function."""

    def test_dangerous_commands_blocked(self):
        assert is_command_blocked("rm -rf /") is True
        assert is_command_blocked("rm -rf /*") is True
        assert is_command_blocked("mkfs /dev/sda") is True

    def test_safe_commands_allowed(self):
        assert is_command_blocked("ls -la") is False
        assert is_command_blocked("echo hello") is False
        assert is_command_blocked("pwd") is False

    def test_case_insensitive(self):
        assert is_command_blocked("RM -RF /") is True


class TestBuiltinTools:
    """Tests for built-in tools."""

    def test_builtin_tools_registered(self):
        expected_tools = [
            "web_search",
            "get_current_time",
            "run_command",
            "system_info",
            "weather",
            "calculator",
            "read_file",
            "list_directory",
            "wikipedia",
            "ip_info",
        ]
        for tool_name in expected_tools:
            assert tool_name in TOOLS, f"Missing builtin tool: {tool_name}"

    def test_tools_have_required_keys(self):
        for name, tool in TOOLS.items():
            assert "func" in tool, f"Tool {name} missing 'func'"
            assert "description" in tool, f"Tool {name} missing 'description'"
            assert callable(tool["func"]), f"Tool {name} 'func' not callable"


class TestGetCurrentTime:
    """Tests for get_current_time tool."""

    def test_returns_string(self):
        result = TOOLS["get_current_time"]["func"]()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_date_format(self):
        result = TOOLS["get_current_time"]["func"]()
        # Should contain year-month-day format
        assert "-" in result


class TestCalculator:
    """Tests for calculator tool."""

    def test_basic_math(self):
        calc = TOOLS["calculator"]["func"]
        assert calc("2 + 2") == "4"
        assert calc("10 * 5") == "50"
        assert calc("100 / 4") == "25.0"

    def test_math_functions(self):
        calc = TOOLS["calculator"]["func"]
        assert calc("sqrt(16)") == "4.0"
        assert calc("abs(-5)") == "5"

    def test_invalid_expression(self):
        calc = TOOLS["calculator"]["func"]
        result = calc("invalid")
        assert "ERROR" in result

    def test_blocked_characters(self):
        calc = TOOLS["calculator"]["func"]
        result = calc("__import__('os')")
        assert "ERROR" in result


class TestReadFile:
    """Tests for read_file tool."""

    def test_nonexistent_file(self):
        result = TOOLS["read_file"]["func"]("/nonexistent/file/path")
        assert "ERROR" in result
        assert "not found" in result.lower()


class TestListDirectory:
    """Tests for list_directory tool."""

    def test_list_current_directory(self):
        result = TOOLS["list_directory"]["func"](".")
        assert isinstance(result, str)

    def test_nonexistent_directory(self):
        result = TOOLS["list_directory"]["func"]("/nonexistent/path")
        assert "ERROR" in result


class TestToolRegistration:
    """Tests for tool registration functions."""

    def test_register_tool_decorator(self):
        # Clean up if exists from previous test
        if "test_decorator_tool" in _TOOLS:
            del _TOOLS["test_decorator_tool"]

        @register_tool("test_decorator_tool", description="Test tool")
        def my_test_tool(x: str) -> str:
            return f"Result: {x}"

        assert "test_decorator_tool" in _TOOLS
        assert _TOOLS["test_decorator_tool"]["func"]("hello") == "Result: hello"

        # Cleanup
        del _TOOLS["test_decorator_tool"]

    def test_register_tool_func(self):
        # Clean up if exists
        if "test_func_tool" in _TOOLS:
            del _TOOLS["test_func_tool"]

        def my_func(x: str) -> str:
            return x.upper()

        register_tool_func("test_func_tool", my_func, "Uppercase tool")
        assert "test_func_tool" in _TOOLS
        assert _TOOLS["test_func_tool"]["func"]("hello") == "HELLO"

        # Cleanup
        del _TOOLS["test_func_tool"]

    def test_register_duplicate_raises_error(self):
        if "duplicate_tool" in _TOOLS:
            del _TOOLS["duplicate_tool"]

        register_tool_func("duplicate_tool", lambda x: x, "First")

        with pytest.raises(ToolRegistrationError):
            register_tool_func("duplicate_tool", lambda x: x, "Second")

        # Cleanup
        del _TOOLS["duplicate_tool"]

    def test_unregister_tool(self):
        if "to_remove" in _TOOLS:
            del _TOOLS["to_remove"]

        register_tool_func("to_remove", lambda: "test", "To be removed")
        assert "to_remove" in _TOOLS

        unregister_tool("to_remove")
        assert "to_remove" not in _TOOLS

    def test_unregister_nonexistent_raises_error(self):
        with pytest.raises(ToolNotFoundError):
            unregister_tool("nonexistent_tool_xyz")


class TestToolQuery:
    """Tests for tool query functions."""

    def test_get_tool(self):
        tool = get_tool("calculator")
        assert "func" in tool
        assert "description" in tool

    def test_get_nonexistent_tool_raises_error(self):
        with pytest.raises(ToolNotFoundError):
            get_tool("nonexistent_tool")

    def test_get_all_tools(self):
        tools = get_all_tools()
        assert isinstance(tools, dict)
        assert "calculator" in tools

    def test_list_tools(self):
        tool_names = list_tools()
        assert isinstance(tool_names, list)
        assert "calculator" in tool_names
        assert "web_search" in tool_names

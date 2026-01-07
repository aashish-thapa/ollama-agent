"""Ollama Agent - A lightweight AI agent library powered by Ollama.

This library provides an easy-to-use agent that can call tools to perform
various tasks like web searches, shell commands, file operations, and more.

Basic usage:
    >>> from ollama_agent import OllamaAgent
    >>> agent = OllamaAgent()
    >>> response = agent.run("What's the weather in NYC?")
    >>> print(response)

Custom tools:
    >>> from ollama_agent import OllamaAgent, register_tool
    >>>
    >>> @register_tool("greet", description="Greet someone by name")
    ... def greet(name: str) -> str:
    ...     return f"Hello, {name}!"
    >>>
    >>> agent = OllamaAgent()
    >>> response = agent.run("Greet John")

Configuration:
    >>> from ollama_agent import OllamaAgent
    >>> agent = OllamaAgent(
    ...     model="llama3.2",
    ...     temperature=0.5,
    ...     approval_callback=lambda tool, input: True
    ... )
"""

__version__ = "0.1.0"

from .agent import OllamaAgent
from .config import Config, config
from .exceptions import (
    ApprovalDeniedError,
    ConfigurationError,
    OllamaAgentError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistrationError,
)
from .tools import (
    TOOLS,
    get_all_tools,
    get_approval_type,
    get_tool,
    is_command_blocked,
    list_tools,
    register_tool,
    register_tool_func,
    unregister_tool,
)

__all__ = [
    # Version
    "__version__",
    # Main class
    "OllamaAgent",
    # Configuration
    "Config",
    "config",
    # Tool registry
    "TOOLS",
    "register_tool",
    "register_tool_func",
    "unregister_tool",
    "get_tool",
    "get_all_tools",
    "list_tools",
    "get_approval_type",
    "is_command_blocked",
    # Exceptions
    "OllamaAgentError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "ToolRegistrationError",
    "ConfigurationError",
    "ApprovalDeniedError",
]

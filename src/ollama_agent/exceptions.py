"""Custom exceptions for ollama-agent."""


class OllamaAgentError(Exception):
    """Base exception for ollama-agent."""

    pass


class ToolNotFoundError(OllamaAgentError):
    """Raised when a tool is not found in the registry."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool not found: {tool_name}")


class ToolExecutionError(OllamaAgentError):
    """Raised when a tool execution fails."""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class ToolRegistrationError(OllamaAgentError):
    """Raised when tool registration fails."""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Failed to register tool '{tool_name}': {message}")


class ConfigurationError(OllamaAgentError):
    """Raised when there's a configuration error."""

    pass


class ApprovalDeniedError(OllamaAgentError):
    """Raised when user denies tool execution."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Execution of tool '{tool_name}' was denied by user")

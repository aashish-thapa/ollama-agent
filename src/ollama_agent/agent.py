"""Core agent module for ollama-agent."""

from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from .config import Config, config as default_config
from .exceptions import ApprovalDeniedError, ToolExecutionError, ToolNotFoundError
from .tools import (
    TOOLS,
    get_all_tools,
    get_approval_type,
    register_tool_func,
    unregister_tool,
)


def _build_system_prompt(tools: Dict[str, Dict[str, Any]]) -> str:
    """Build system prompt with all available tools.

    Args:
        tools: Dictionary of tools to include in the prompt

    Returns:
        Formatted system prompt string
    """
    tool_list = "\n".join(
        f"- {name}: {info['description']}" for name, info in tools.items()
    )

    return f"""You are a helpful assistant with access to various tools.

AVAILABLE TOOLS:
{tool_list}

HOW TO USE TOOLS:
When you need to use a tool, respond ONLY with:
TOOL: tool_name
INPUT: the input for the tool (leave empty if not needed)

RULES:
1. Use tools when you need real-time data, system info, or to perform actions
2. After receiving tool results, provide a helpful summary
3. For commands, be specific about what you're running
4. Always explain what you're doing before using a tool
5. If a tool fails, explain the error and suggest alternatives"""


class OllamaAgent:
    """AI agent powered by Ollama with tool-calling capabilities.

    This agent uses a local Ollama model to process queries and can call
    various tools to perform actions like web searches, shell commands,
    file operations, and more.

    Example:
        >>> from ollama_agent import OllamaAgent
        >>> agent = OllamaAgent()
        >>> response = agent.run("What's the current time?")
        >>> print(response)

    With custom approval callback:
        >>> def my_approval(tool_name: str, tool_input: str) -> bool:
        ...     return input(f"Allow {tool_name}? (y/n): ").lower() == 'y'
        >>> agent = OllamaAgent(approval_callback=my_approval)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_iterations: Optional[int] = None,
        approval_callback: Optional[Callable[[str, str], bool]] = None,
        tools: Optional[Dict[str, Dict[str, Any]]] = None,
        config: Optional[Config] = None,
    ):
        """Initialize the Ollama agent.

        Args:
            model: Ollama model name (default: from config or "llama3.2")
            base_url: Ollama API URL (default: from config or "http://localhost:11434")
            temperature: Model temperature 0-1 (default: from config or 0.7)
            max_iterations: Max tool calls per query (default: from config or 10)
            approval_callback: Optional function(tool_name, tool_input) -> bool
                              Returns True if approved, False if denied.
                              If None, tools run without approval.
            tools: Optional custom tools dictionary. If None, uses global TOOLS.
            config: Optional Config instance. If None, uses default config.
        """
        self._config = config or default_config

        # Override config with explicit parameters
        self._model = model or self._config.ollama_model
        self._base_url = base_url or self._config.ollama_base_url
        self._temperature = temperature if temperature is not None else self._config.temperature
        self._max_iterations = max_iterations or self._config.max_iterations

        self.llm = ChatOllama(
            model=self._model,
            base_url=self._base_url,
            temperature=self._temperature,
        )

        self.approval_callback = approval_callback
        self._tools = tools if tools is not None else TOOLS
        self._messages: List = []
        self._rebuild_system_prompt()

    def _rebuild_system_prompt(self) -> None:
        """Rebuild the system prompt with current tools."""
        self._messages = [SystemMessage(content=_build_system_prompt(self._tools))]

    @property
    def tools(self) -> Dict[str, Dict[str, Any]]:
        """Get the current tools dictionary."""
        return self._tools.copy()

    @property
    def model(self) -> str:
        """Get the current model name."""
        return self._model

    def add_tool(
        self,
        name: str,
        func: Callable,
        description: str,
        requires_approval: Optional[str] = None,
    ) -> None:
        """Add a custom tool to this agent instance.

        Args:
            name: Tool name
            func: The function to call
            description: Description for the LLM
            requires_approval: Optional approval type ("commands" or "files")

        Raises:
            ToolRegistrationError: If tool already exists
        """
        if self._tools is TOOLS:
            # Using global tools, need to register globally
            register_tool_func(name, func, description, requires_approval)
        else:
            # Using custom tools dict
            self._tools[name] = {"func": func, "description": description}

        self._rebuild_system_prompt()

    def remove_tool(self, name: str) -> None:
        """Remove a tool from this agent instance.

        Args:
            name: Tool name to remove

        Raises:
            ToolNotFoundError: If tool doesn't exist
        """
        if name not in self._tools:
            raise ToolNotFoundError(name)

        if self._tools is TOOLS:
            unregister_tool(name)
        else:
            del self._tools[name]

        self._rebuild_system_prompt()

    def _parse_tool_call(self, response: str) -> Optional[Tuple[str, str]]:
        """Parse tool call from response.

        Args:
            response: LLM response text

        Returns:
            Tuple of (tool_name, tool_input) or None if no tool call
        """
        lines = response.strip().split("\n")
        tool_name = None
        tool_input = ""

        for line in lines:
            if line.startswith("TOOL:"):
                tool_name = line.replace("TOOL:", "").strip()
            elif line.startswith("INPUT:"):
                tool_input = line.replace("INPUT:", "").strip()

        if tool_name and tool_name in self._tools:
            return (tool_name, tool_input)
        return None

    def _needs_approval(self, tool_name: str) -> bool:
        """Check if a tool needs user approval.

        Args:
            tool_name: Name of the tool

        Returns:
            True if approval is needed
        """
        approval_type = get_approval_type(tool_name)
        if not approval_type:
            return False

        if approval_type == "commands":
            return self._config.require_approval_commands
        elif approval_type == "files":
            return self._config.require_approval_files
        return False

    def _execute_tool(
        self, tool_name: str, tool_input: str
    ) -> Tuple[str, bool]:
        """Execute a tool and return (result, was_executed).

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input to pass to the tool

        Returns:
            Tuple of (result_string, was_executed_bool)
        """
        tool_info = self._tools.get(tool_name)
        if not tool_info:
            return f"Unknown tool: {tool_name}", False

        # Check approval
        if self._needs_approval(tool_name) and self.approval_callback:
            if not self.approval_callback(tool_name, tool_input):
                return "Tool execution denied by user.", False

        try:
            func = tool_info["func"]
            if tool_input:
                result = func(tool_input)
            else:
                result = func()
            return result, True
        except Exception as e:
            return f"Tool error: {e}", False

    def run(
        self,
        query: str,
        verbose: bool = False,
    ) -> str:
        """Run the agent with a query and return the response.

        Args:
            query: User query/prompt
            verbose: If True, print tool execution info

        Returns:
            Final response string from the agent

        Example:
            >>> agent = OllamaAgent()
            >>> response = agent.run("What's 2 + 2?")
            >>> print(response)
        """
        self._messages.append(HumanMessage(content=query))

        for _ in range(self._max_iterations):
            try:
                response = self.llm.invoke(self._messages)
                response_text = response.content

                tool_call = self._parse_tool_call(response_text)

                if tool_call:
                    tool_name, tool_input = tool_call

                    if verbose:
                        print(f"\n[Tool: {tool_name}]")
                        if tool_input:
                            print(f"[Input: {tool_input}]")

                    result, executed = self._execute_tool(tool_name, tool_input)

                    if verbose:
                        print(f"[{'Done' if executed else 'Skipped'}]\n")

                    self._messages.append(AIMessage(content=response_text))
                    self._messages.append(
                        HumanMessage(
                            content=f"TOOL RESULT:\n{result}\n\nProvide a helpful response based on this result."
                        )
                    )
                else:
                    self._messages.append(AIMessage(content=response_text))
                    return response_text

            except Exception as e:
                return f"Error: {e}"

        return "Max iterations reached."

    def reset(self) -> None:
        """Reset conversation history, keeping the system prompt."""
        self._rebuild_system_prompt()

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history as a list of message dicts.

        Returns:
            List of {"role": str, "content": str} dicts
        """
        history = []
        for msg in self._messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "unknown"
            history.append({"role": role, "content": msg.content})
        return history

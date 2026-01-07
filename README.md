# Ollama Agent

[![PyPI version](https://img.shields.io/pypi/v/ollama-agent.svg)](https://pypi.org/project/ollama-agent/)
[![Python Versions](https://img.shields.io/pypi/pyversions/ollama-agent.svg)](https://pypi.org/project/ollama-agent/)
[![CI](https://github.com/aashish-thapa/ollama-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/aashish-thapa/ollama-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/ollama-agent.svg)](https://pypi.org/project/ollama-agent/)

A lightweight AI agent library powered by [Ollama](https://ollama.ai) with built-in tools and custom tool support.

## Features

- **Simple API** - Get started with just a few lines of code
- **10 Built-in Tools** - Web search, calculator, weather, system info, and more
- **Custom Tools** - Easy registration via decorators or functions
- **Configurable** - Environment variables or constructor parameters
- **Safe by Default** - Optional user approval for dangerous operations
- **Conversation Memory** - Maintains context across queries

## Installation

```bash
pip install ollama-agent
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- A model pulled (e.g., `ollama pull llama3.2`)

## Quick Start

```python
from ollama_agent import OllamaAgent

# Create an agent
agent = OllamaAgent()

# Run a query
response = agent.run("What's the current time?")
print(response)

# Conversation history is maintained
response = agent.run("And what's the weather in New York?")
print(response)

# Reset when needed
agent.reset()
```

## Custom Tools

### Using Decorators

```python
from ollama_agent import OllamaAgent, register_tool

@register_tool("greet", description="Greet someone by name. Input: name")
def greet(name: str) -> str:
    return f"Hello, {name}!"

agent = OllamaAgent()
response = agent.run("Greet Alice")
```

### Using Function Registration

```python
from ollama_agent import register_tool_func

def my_tool(input_str: str) -> str:
    return f"Processed: {input_str}"

register_tool_func(
    name="my_tool",
    func=my_tool,
    description="Process input text"
)
```

### Instance-specific Tools

```python
agent = OllamaAgent()

agent.add_tool(
    name="uppercase",
    func=lambda text: text.upper(),
    description="Convert text to uppercase"
)

response = agent.run("Convert 'hello' to uppercase")
agent.remove_tool("uppercase")
```

## Configuration

### Constructor Parameters

```python
agent = OllamaAgent(
    model="llama3.2",                    # Ollama model name
    base_url="http://localhost:11434",   # Ollama API URL
    temperature=0.7,                     # Model temperature (0-1)
    max_iterations=10,                   # Max tool calls per query
    approval_callback=my_callback,       # Optional approval function
)
```

### Environment Variables

```bash
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
TEMPERATURE=0.7
MAX_ITERATIONS=10
MAX_SEARCH_RESULTS=5
REQUIRE_APPROVAL_COMMANDS=true
REQUIRE_APPROVAL_FILES=false
```

## Approval Callback

Require user approval for dangerous operations:

```python
def approval_callback(tool_name: str, tool_input: str) -> bool:
    print(f"Tool: {tool_name}, Input: {tool_input}")
    return input("Allow? (y/n): ").lower() == "y"

agent = OllamaAgent(approval_callback=approval_callback)
```

## Built-in Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web using DuckDuckGo |
| `get_current_time` | Get current date and time |
| `run_command` | Run shell commands (requires approval) |
| `system_info` | Get CPU, memory, disk, uptime info |
| `weather` | Get current weather |
| `calculator` | Evaluate math expressions |
| `read_file` | Read file contents |
| `list_directory` | List directory contents |
| `wikipedia` | Search Wikipedia |
| `ip_info` | Get public IP and location |

## API Reference

### OllamaAgent

```python
class OllamaAgent:
    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        temperature: float = None,
        max_iterations: int = None,
        approval_callback: Callable[[str, str], bool] = None,
        tools: dict = None,
        config: Config = None,
    ): ...

    def run(self, query: str, verbose: bool = False) -> str: ...
    def reset(self) -> None: ...
    def add_tool(self, name, func, description, requires_approval=None) -> None: ...
    def remove_tool(self, name: str) -> None: ...
    def get_history(self) -> list[dict]: ...

    @property
    def tools(self) -> dict: ...
    @property
    def model(self) -> str: ...
```

### Tool Functions

```python
# Decorator registration
@register_tool(name: str, description: str, requires_approval: str = None)
def my_tool(input: str) -> str: ...

# Function registration
register_tool_func(name, func, description, requires_approval=None)

# Management
unregister_tool(name: str)
get_tool(name: str) -> dict
get_all_tools() -> dict
list_tools() -> list[str]
```

## Development

```bash
# Clone the repo
git clone https://github.com/aashish-thapa/ollama-agent.git
cd ollama-agent

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

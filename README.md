# Ollama Agent

A lightweight AI agent library powered by Ollama with built-in tools and custom tool support.

## Features

- Simple, intuitive API
- 10 built-in tools (web search, calculator, weather, system info, etc.)
- Easy custom tool registration via decorators or functions
- Configurable via environment variables or constructor parameters
- Optional user approval for dangerous operations
- Conversation history management

## Installation

```bash
pip install ollama-agent
```

Or install from source:

```bash
git clone https://github.com/aashish-thapa/ollama-agent.git
cd ollama-agent-lib
pip install -e .
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running locally
- A model pulled (e.g., `ollama pull llama3.2`)

## Quick Start

```python
from ollama_agent import OllamaAgent

# Create an agent
agent = OllamaAgent()

# Run a query
response = agent.run("What's the current time?")
print(response)

# Run another query (conversation history is maintained)
response = agent.run("And what's the weather in New York?")
print(response)

# Reset conversation when needed
agent.reset()
```

## Custom Tools

### Using Decorators

```python
from ollama_agent import OllamaAgent, register_tool

@register_tool("greet", description="Greet someone by name. Input: name")
def greet(name: str) -> str:
    return f"Hello, {name}! Nice to meet you."

@register_tool("add_numbers", description="Add two numbers. Input: two numbers separated by comma")
def add_numbers(input_str: str) -> str:
    a, b = map(float, input_str.split(","))
    return f"Result: {a + b}"

agent = OllamaAgent()
response = agent.run("Greet Alice")
```

### Using Function Registration

```python
from ollama_agent import OllamaAgent, register_tool_func

def my_tool(input_str: str) -> str:
    return f"Processed: {input_str}"

register_tool_func(
    name="my_tool",
    func=my_tool,
    description="Process some input. Input: text to process"
)

agent = OllamaAgent()
```

### Instance-specific Tools

```python
from ollama_agent import OllamaAgent

agent = OllamaAgent()

# Add tool to this agent only
agent.add_tool(
    name="uppercase",
    func=lambda text: text.upper(),
    description="Convert text to uppercase. Input: text"
)

response = agent.run("Convert 'hello' to uppercase")

# Remove when done
agent.remove_tool("uppercase")
```

## Configuration

### Via Constructor

```python
from ollama_agent import OllamaAgent

agent = OllamaAgent(
    model="llama3.2",              # Ollama model name
    base_url="http://localhost:11434",  # Ollama API URL
    temperature=0.7,               # Model temperature (0-1)
    max_iterations=10,             # Max tool calls per query
    approval_callback=my_callback, # Optional approval function
)
```

### Via Environment Variables

```bash
export OLLAMA_MODEL=llama3.2
export OLLAMA_BASE_URL=http://localhost:11434
export TEMPERATURE=0.7
export MAX_ITERATIONS=10
export MAX_SEARCH_RESULTS=5
export REQUIRE_APPROVAL_COMMANDS=true
export REQUIRE_APPROVAL_FILES=false
```

## Approval Callback

For dangerous operations (like shell commands), you can require user approval:

```python
from ollama_agent import OllamaAgent

def approval_callback(tool_name: str, tool_input: str) -> bool:
    """Return True to allow, False to deny."""
    print(f"Tool: {tool_name}")
    print(f"Input: {tool_input}")
    return input("Allow? (y/n): ").lower() == "y"

agent = OllamaAgent(approval_callback=approval_callback)
response = agent.run("List files in my home directory")
```

## Built-in Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web using DuckDuckGo |
| `get_current_time` | Get current date and time |
| `run_command` | Run shell commands (requires approval by default) |
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
    def add_tool(self, name: str, func: Callable, description: str, requires_approval: str = None) -> None: ...
    def remove_tool(self, name: str) -> None: ...
    def get_history(self) -> list[dict]: ...
```

### Tool Registration

```python
# Decorator
@register_tool(name: str, description: str, requires_approval: str = None)
def my_tool(input: str) -> str: ...

# Function
register_tool_func(name: str, func: Callable, description: str, requires_approval: str = None)

# Unregister
unregister_tool(name: str)

# Query
get_all_tools() -> dict
list_tools() -> list[str]
get_tool(name: str) -> dict
```

## License

MIT License - see [LICENSE](LICENSE) for details.

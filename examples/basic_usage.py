"""Basic usage example for ollama-agent."""

from ollama_agent import OllamaAgent


def simple_example():
    """Simple agent usage without approval callback."""
    # Create agent with default settings
    agent = OllamaAgent()

    # Run a simple query
    response = agent.run("What's the current time?")
    print("Response:", response)

    # Run another query (conversation history is maintained)
    response = agent.run("And what day of the week is it?")
    print("Response:", response)

    # Reset conversation history if needed
    agent.reset()


def with_approval_callback():
    """Agent with user approval for dangerous operations."""

    def approval_callback(tool_name: str, tool_input: str) -> bool:
        """Ask user for approval before running certain tools."""
        print(f"\n[Approval Required]")
        print(f"Tool: {tool_name}")
        print(f"Input: {tool_input}")
        answer = input("Allow this operation? (y/n): ").strip().lower()
        return answer == "y"

    # Create agent with approval callback
    agent = OllamaAgent(approval_callback=approval_callback)

    # This will trigger approval for run_command tool
    response = agent.run("List the files in my home directory", verbose=True)
    print("Response:", response)


def with_custom_config():
    """Agent with custom configuration."""
    # Create agent with custom settings
    agent = OllamaAgent(
        model="llama3.2",  # Specific model
        temperature=0.5,  # Lower temperature for more focused responses
        max_iterations=5,  # Limit tool call iterations
    )

    response = agent.run("Calculate the square root of 144")
    print("Response:", response)


def verbose_mode():
    """Show tool execution details."""
    agent = OllamaAgent()

    # Enable verbose mode to see tool calls
    response = agent.run("Search for Python programming tutorials", verbose=True)
    print("Response:", response)


if __name__ == "__main__":
    print("=" * 50)
    print("Basic Usage Example")
    print("=" * 50)
    simple_example()

    print("\n" + "=" * 50)
    print("Custom Config Example")
    print("=" * 50)
    with_custom_config()

    print("\n" + "=" * 50)
    print("Verbose Mode Example")
    print("=" * 50)
    verbose_mode()

    # Uncomment to test approval callback
    # print("\n" + "=" * 50)
    # print("Approval Callback Example")
    # print("=" * 50)
    # with_approval_callback()

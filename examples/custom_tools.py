"""Custom tools example for ollama-agent."""

import random

from ollama_agent import OllamaAgent, register_tool, register_tool_func, list_tools


# Method 1: Using the decorator
@register_tool("roll_dice", description="Roll a dice. Input: number of sides (default: 6)")
def roll_dice(sides: str = "6") -> str:
    """Roll a dice with the specified number of sides."""
    try:
        num_sides = int(sides) if sides else 6
        result = random.randint(1, num_sides)
        return f"Rolled a {num_sides}-sided dice: {result}"
    except ValueError:
        return f"Invalid number of sides: {sides}"


@register_tool("coin_flip", description="Flip a coin. No input needed.")
def coin_flip() -> str:
    """Flip a coin."""
    result = random.choice(["Heads", "Tails"])
    return f"Coin flip result: {result}"


@register_tool(
    "word_count",
    description="Count words in text. Input: text to count words in",
)
def word_count(text: str) -> str:
    """Count words in the given text."""
    if not text:
        return "No text provided"
    words = len(text.split())
    chars = len(text)
    return f"Word count: {words}, Character count: {chars}"


def example_decorator_tools():
    """Example using tools registered with decorators."""
    print("Registered tools:", list_tools())

    agent = OllamaAgent()

    # Test custom tools
    print("\n--- Roll Dice ---")
    response = agent.run("Roll a 20-sided dice for me", verbose=True)
    print("Response:", response)

    agent.reset()

    print("\n--- Coin Flip ---")
    response = agent.run("Flip a coin", verbose=True)
    print("Response:", response)

    agent.reset()

    print("\n--- Word Count ---")
    response = agent.run(
        "Count the words in: The quick brown fox jumps over the lazy dog",
        verbose=True,
    )
    print("Response:", response)


def example_function_registration():
    """Example using register_tool_func for dynamic registration."""

    # Define a tool function
    def reverse_text(text: str) -> str:
        """Reverse the given text."""
        return f"Reversed: {text[::-1]}"

    # Register it dynamically
    register_tool_func(
        name="reverse_text",
        func=reverse_text,
        description="Reverse text. Input: text to reverse",
    )

    agent = OllamaAgent()

    response = agent.run("Reverse the text: Hello World", verbose=True)
    print("Response:", response)


def example_instance_tools():
    """Example adding tools to a specific agent instance."""
    agent = OllamaAgent()

    # Add a tool to just this agent instance
    def uppercase(text: str) -> str:
        return text.upper()

    agent.add_tool(
        name="uppercase",
        func=uppercase,
        description="Convert text to uppercase. Input: text",
    )

    response = agent.run("Convert 'hello world' to uppercase", verbose=True)
    print("Response:", response)

    # Remove the tool
    agent.remove_tool("uppercase")
    print("\nTool removed. Current tools:", list(agent.tools.keys()))


if __name__ == "__main__":
    print("=" * 50)
    print("Custom Tools with Decorators")
    print("=" * 50)
    example_decorator_tools()

    print("\n" + "=" * 50)
    print("Dynamic Tool Registration")
    print("=" * 50)
    example_function_registration()

    print("\n" + "=" * 50)
    print("Instance-specific Tools")
    print("=" * 50)
    example_instance_tools()

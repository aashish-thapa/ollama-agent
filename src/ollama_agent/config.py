"""Configuration module for ollama-agent."""

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _parse_bool(value: str | None, default: bool = False) -> bool:
    """Parse a boolean from environment variable string."""
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes")


@dataclass
class Config:
    """Configuration for the Ollama agent.

    All settings can be overridden via environment variables or
    by passing them directly to the OllamaAgent constructor.

    Environment variables:
        OLLAMA_MODEL: Model name (default: llama3.2)
        OLLAMA_BASE_URL: Ollama API URL (default: http://localhost:11434)
        TEMPERATURE: Model temperature 0-1 (default: 0.7)
        MAX_ITERATIONS: Max tool calls per query (default: 10)
        MAX_SEARCH_RESULTS: Max web search results (default: 5)
        REQUIRE_APPROVAL_COMMANDS: Require approval for shell commands (default: true)
        REQUIRE_APPROVAL_FILES: Require approval for file writes (default: false)
    """

    # Ollama settings
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "10"))

    # Tool settings
    max_search_results: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))

    # Approval settings
    require_approval_commands: bool = _parse_bool(
        os.getenv("REQUIRE_APPROVAL_COMMANDS", "true"), True
    )
    require_approval_files: bool = _parse_bool(
        os.getenv("REQUIRE_APPROVAL_FILES", "false"), False
    )

    # Safety settings
    blocked_commands: List[str] = field(
        default_factory=lambda: [
            "rm -rf /",
            "rm -rf /*",
            "mkfs",
            "dd if=",
            ":(){:|:&};:",
            "chmod -R 777 /",
            "chown -R",
            "> /dev/sda",
            "wget | sh",
            "curl | sh",
        ]
    )


# Global config instance
config = Config()

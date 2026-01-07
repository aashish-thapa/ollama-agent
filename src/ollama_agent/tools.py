"""Tool registry and built-in tools for ollama-agent."""

import json
import math
import os
import subprocess
import urllib.error
import urllib.request
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from duckduckgo_search import DDGS

from .config import config
from .exceptions import ToolNotFoundError, ToolRegistrationError

# Tool metadata for approval requirements
_TOOLS_REQUIRING_APPROVAL: Dict[str, str] = {
    "run_command": "commands",
    "write_file": "files",
}

# Global tool registry
_TOOLS: Dict[str, Dict[str, Any]] = {}


def get_approval_type(tool_name: str) -> Optional[str]:
    """Return the approval type needed for a tool, or None if no approval needed.

    Args:
        tool_name: Name of the tool

    Returns:
        Approval type string ("commands" or "files") or None
    """
    return _TOOLS_REQUIRING_APPROVAL.get(tool_name)


def is_command_blocked(command: str) -> bool:
    """Check if a command is in the blocked list.

    Args:
        command: The shell command to check

    Returns:
        True if the command is blocked
    """
    cmd_lower = command.lower()
    return any(blocked in cmd_lower for blocked in config.blocked_commands)


def register_tool(
    name: str,
    description: str,
    requires_approval: Optional[str] = None,
) -> Callable:
    """Decorator to register a function as a tool.

    Args:
        name: Tool name (used in TOOL: calls)
        description: Description shown to the LLM
        requires_approval: Optional approval type ("commands" or "files")

    Returns:
        Decorator function

    Example:
        @register_tool("my_tool", description="Does something useful")
        def my_tool(input_str: str) -> str:
            return f"Result: {input_str}"
    """

    def decorator(func: Callable) -> Callable:
        if name in _TOOLS:
            raise ToolRegistrationError(name, "Tool already exists")

        _TOOLS[name] = {
            "func": func,
            "description": description,
        }

        if requires_approval:
            _TOOLS_REQUIRING_APPROVAL[name] = requires_approval

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def register_tool_func(
    name: str,
    func: Callable,
    description: str,
    requires_approval: Optional[str] = None,
) -> None:
    """Register a function as a tool (non-decorator version).

    Args:
        name: Tool name
        func: The function to register
        description: Description shown to the LLM
        requires_approval: Optional approval type ("commands" or "files")

    Raises:
        ToolRegistrationError: If tool already exists
    """
    if name in _TOOLS:
        raise ToolRegistrationError(name, "Tool already exists")

    _TOOLS[name] = {
        "func": func,
        "description": description,
    }

    if requires_approval:
        _TOOLS_REQUIRING_APPROVAL[name] = requires_approval


def unregister_tool(name: str) -> None:
    """Remove a tool from the registry.

    Args:
        name: Tool name to remove

    Raises:
        ToolNotFoundError: If tool doesn't exist
    """
    if name not in _TOOLS:
        raise ToolNotFoundError(name)

    del _TOOLS[name]
    _TOOLS_REQUIRING_APPROVAL.pop(name, None)


def get_tool(name: str) -> Dict[str, Any]:
    """Get a tool by name.

    Args:
        name: Tool name

    Returns:
        Tool dictionary with "func" and "description"

    Raises:
        ToolNotFoundError: If tool doesn't exist
    """
    if name not in _TOOLS:
        raise ToolNotFoundError(name)
    return _TOOLS[name]


def get_all_tools() -> Dict[str, Dict[str, Any]]:
    """Get all registered tools.

    Returns:
        Dictionary of all tools
    """
    return _TOOLS.copy()


def list_tools() -> list[str]:
    """Get list of all tool names.

    Returns:
        List of tool names
    """
    return list(_TOOLS.keys())


# ============ BUILT-IN TOOLS ============


def _web_search(query: str) -> str:
    """Search the web using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=config.max_search_results))

        if not results:
            return "No results found."

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"{i}. {r.get('title', 'No title')}\n"
                f"   URL: {r.get('href', '')}\n"
                f"   {r.get('body', '')}"
            )
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search failed: {e}"


def _get_current_time() -> str:
    """Get current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")


def _run_command(command: str) -> str:
    """Execute a shell command and return output."""
    if is_command_blocked(command):
        return "ERROR: This command is blocked for safety reasons."

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=Path.home(),
        )
        output = result.stdout or result.stderr or "Command executed (no output)"
        if len(output) > 2000:
            output = output[:2000] + "\n... (truncated)"
        return output
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out (30s limit)"
    except Exception as e:
        return f"ERROR: {e}"


def _system_info() -> str:
    """Get system information (CPU, memory, disk)."""
    info = []

    # Uptime
    try:
        with open("/proc/uptime") as f:
            uptime_seconds = float(f.read().split()[0])
            days, rem = divmod(int(uptime_seconds), 86400)
            hours, rem = divmod(rem, 3600)
            mins, _ = divmod(rem, 60)
            info.append(f"Uptime: {days}d {hours}h {mins}m")
    except Exception:
        pass

    # CPU
    try:
        with open("/proc/loadavg") as f:
            load = f.read().split()[:3]
            info.append(f"Load average: {' '.join(load)}")
    except Exception:
        pass

    # Memory
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                parts = line.split()
                if parts[0] in ("MemTotal:", "MemAvailable:", "MemFree:"):
                    mem[parts[0][:-1]] = int(parts[1]) // 1024  # MB

            total = mem.get("MemTotal", 0)
            avail = mem.get("MemAvailable", mem.get("MemFree", 0))
            used = total - avail
            pct = (used / total * 100) if total else 0
            info.append(f"Memory: {used}MB / {total}MB ({pct:.1f}% used)")
    except Exception:
        pass

    # Disk
    try:
        stat = os.statvfs("/")
        total = stat.f_blocks * stat.f_frsize // (1024**3)
        free = stat.f_bavail * stat.f_frsize // (1024**3)
        used = total - free
        pct = (used / total * 100) if total else 0
        info.append(f"Disk (/): {used}GB / {total}GB ({pct:.1f}% used)")
    except Exception:
        pass

    return "\n".join(info) if info else "Could not retrieve system info"


def _weather(location: str = "") -> str:
    """Get weather using wttr.in (free, no API key)."""
    try:
        loc = location.replace(" ", "+") if location else ""
        url = f"https://wttr.in/{loc}?format=3"
        req = urllib.request.Request(url, headers={"User-Agent": "curl/7.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode().strip()
    except Exception as e:
        return f"Weather fetch failed: {e}"


def _weather_detailed(location: str = "") -> str:
    """Get detailed weather forecast."""
    try:
        loc = location.replace(" ", "+") if location else ""
        url = f"https://wttr.in/{loc}?format=%l:+%c+%t+%h+%w"
        req = urllib.request.Request(url, headers={"User-Agent": "curl/7.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode().strip()
    except Exception as e:
        return f"Weather fetch failed: {e}"


def _calculator(expression: str) -> str:
    """Evaluate a math expression safely."""
    allowed_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "pi": math.pi,
        "e": math.e,
        "floor": math.floor,
        "ceil": math.ceil,
    }

    try:
        # Remove dangerous characters
        for char in "_%\\`[]{}":
            if char in expression:
                return f"ERROR: Invalid character '{char}'"

        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"ERROR: {e}"


def _read_file(filepath: str) -> str:
    """Read contents of a file."""
    try:
        path = Path(filepath).expanduser()
        if not path.exists():
            return f"ERROR: File not found: {filepath}"
        if not path.is_file():
            return f"ERROR: Not a file: {filepath}"
        if path.stat().st_size > 100_000:
            return "ERROR: File too large (>100KB)"

        content = path.read_text()
        if len(content) > 3000:
            content = content[:3000] + "\n... (truncated)"
        return content
    except Exception as e:
        return f"ERROR: {e}"


def _list_directory(path: str = ".") -> str:
    """List contents of a directory."""
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"ERROR: Path not found: {path}"
        if not p.is_dir():
            return f"ERROR: Not a directory: {path}"

        items = []
        for item in sorted(p.iterdir())[:50]:  # Limit to 50 items
            prefix = "[DIR] " if item.is_dir() else "[FILE] "
            items.append(f"{prefix}{item.name}")

        return "\n".join(items) if items else "(empty directory)"
    except Exception as e:
        return f"ERROR: {e}"


def _wikipedia(query: str) -> str:
    """Search Wikipedia for a summary."""
    try:
        search_url = (
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        )
        req = urllib.request.Request(
            search_url, headers={"User-Agent": "OllamaAgent/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            title = data.get("title", query)
            extract = data.get("extract", "No summary available.")
            url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
            return f"**{title}**\n\n{extract}\n\nSource: {url}"
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return f"No Wikipedia article found for '{query}'"
        return f"Wikipedia error: {e}"
    except Exception as e:
        return f"Wikipedia error: {e}"


def _ip_info() -> str:
    """Get public IP and basic network info."""
    try:
        req = urllib.request.Request(
            "https://ipinfo.io/json", headers={"User-Agent": "OllamaAgent/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return (
                f"IP: {data.get('ip', 'N/A')}\n"
                f"Location: {data.get('city', '')}, {data.get('region', '')}, {data.get('country', '')}\n"
                f"ISP: {data.get('org', 'N/A')}"
            )
    except Exception as e:
        return f"Could not fetch IP info: {e}"


# Register all built-in tools
def _register_builtin_tools():
    """Register all built-in tools."""
    builtins = [
        ("web_search", _web_search, "Search the web using DuckDuckGo. Input: search query"),
        ("get_current_time", _get_current_time, "Get current date and time. No input needed."),
        (
            "run_command",
            _run_command,
            "Run a shell command on the system. Input: command to run",
        ),
        (
            "system_info",
            _system_info,
            "Get system info (CPU, memory, disk, uptime). No input needed.",
        ),
        (
            "weather",
            _weather,
            "Get current weather. Input: city name (optional, defaults to auto-detect)",
        ),
        (
            "calculator",
            _calculator,
            "Calculate math expressions. Input: expression like '2+2' or 'sqrt(16)'",
        ),
        ("read_file", _read_file, "Read a file's contents. Input: file path"),
        (
            "list_directory",
            _list_directory,
            "List files in a directory. Input: path (default: current dir)",
        ),
        ("wikipedia", _wikipedia, "Search Wikipedia for information. Input: search term"),
        ("ip_info", _ip_info, "Get your public IP and location info. No input needed."),
    ]

    for name, func, description in builtins:
        _TOOLS[name] = {"func": func, "description": description}


# Initialize built-in tools
_register_builtin_tools()

# Public alias for backward compatibility
TOOLS = _TOOLS

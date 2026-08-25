"""
Action Dispatcher — Maps capability tool names to actual action functions.
Single entry point for the Executor to call any tool by its capability name.
"""

from typing import Callable, Any, Dict, Optional
import importlib
import sys
from pathlib import Path

# Ensure project root is in path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Tool name -> (module_path, function_name, required_params)
# This maps the capability "tools" names to actual implementations
TOOL_DISPATCH_TABLE: Dict[str, Dict[str, Any]] = {
    # System Control
    "open_app": {
        "module": "actions.open_app",
        "function": "open_app",
        "params": ["parameters", "player"],
        "returns": "str",
    },
    "computer_settings": {
        "module": "actions.computer_settings",
        "function": "computer_settings",
        "params": ["parameters", "response", "player", "session_memory"],
        "returns": "str",
    },
    "system_volume_up": {
        "module": "actions.computer_settings",
        "function": "computer_settings",
        "params": ["parameters", "response", "player", "session_memory"],
        "returns": "str",
        "inject": {"action": "volume_up"},
    },
    "system_volume_down": {
        "module": "actions.computer_settings",
        "function": "computer_settings",
        "params": ["parameters", "response", "player", "session_memory"],
        "returns": "str",
        "inject": {"action": "volume_down"},
    },
    "system_mute": {
        "module": "actions.computer_settings",
        "function": "computer_settings",
        "params": ["parameters", "response", "player", "session_memory"],
        "returns": "str",
        "inject": {"action": "volume_mute"},
    },
    "system_shutdown": {
        "module": "actions.system_shutdown",
        "function": "system_shutdown",
        "params": ["parameters", "player", "speak"],
        "returns": "str",
    },
    "get_time": {
        "module": "actions.get_time",
        "function": "get_time",
        "params": ["parameters", "player"],
        "returns": "str",
    },

    # Web Actions
    "web_search": {
        "module": "actions.web_search",
        "function": "web_search",
        "params": ["parameters", "player"],
        "returns": "str",
    },
    "weather_report": {
        "module": "actions.weather_report",
        "function": "weather_action",
        "params": ["parameters", "player"],
        "returns": "str",
    },
    "flight_search": {
        "module": "actions.flight_finder",
        "function": "flight_finder",
        "params": ["parameters", "player", "speak"],
        "returns": "str",
    },
    "browser_navigation": {
        "module": "actions.browser_control",
        "function": "browser_control",
        "params": ["parameters", "response", "player", "session_memory"],
        "returns": "str",
    },

    # Media - YouTube (using youtube_video module with action inject)
    "youtube_play": {
        "module": "actions.youtube_video",
        "function": "youtube_video",
        "params": ["parameters", "response", "player", "session_memory"],
        "returns": "str",
        "inject": {"action": "play"},
    },
    "youtube_pause": {
        "module": "actions.youtube_video",
        "function": "youtube_video",
        "params": ["parameters", "response", "player", "session_memory"],
        "returns": "str",
        "inject": {"action": "pause"},
    },
    "youtube_resume": {
        "module": "actions.youtube_video",
        "function": "youtube_video",
        "params": ["parameters", "response", "player", "session_memory"],
        "returns": "str",
        "inject": {"action": "resume"},
    },
    "youtube_volume": {
        "module": "actions.youtube_video",
        "function": "youtube_video",
        "params": ["parameters", "response", "player", "session_memory"],
        "returns": "str",
        "inject": {"action": "volume"},
    },
    "youtube_next": {
        "module": "actions.youtube_video",
        "function": "youtube_video",
        "params": ["parameters", "response", "player", "session_memory"],
        "returns": "str",
        "inject": {"action": "next"},
    },
    "youtube_previous": {
        "module": "actions.youtube_video",
        "function": "youtube_video",
        "params": ["parameters", "response", "player", "session_memory"],
        "returns": "str",
        "inject": {"action": "previous"},
    },

    # File Operations
    "file_operations": {
        "module": "actions.file_controller",
        "function": "file_controller",
        "params": ["parameters", "player"],
        "returns": "str",
    },
    "file_read": {
        "module": "actions.file_controller",
        "function": "file_controller",
        "params": ["parameters", "player"],
        "returns": "str",
        "inject": {"action": "read"},
    },
    "file_write": {
        "module": "actions.file_controller",
        "function": "file_controller",
        "params": ["parameters", "player"],
        "returns": "str",
        "inject": {"action": "write"},
    },
    "file_list": {
        "module": "actions.file_controller",
        "function": "file_controller",
        "params": ["parameters", "player"],
        "returns": "str",
        "inject": {"action": "list"},
    },

    # Desktop Management
    "desktop_management": {
        "module": "actions.desktop",
        "function": "desktop_control",
        "params": ["parameters", "response", "player", "session_memory"],
        "returns": "str",
    },

    # Coding / Development
    "code_assistance": {
        "module": "actions.code_helper",
        "function": "code_helper",
        "params": ["parameters", "response", "player", "session_memory", "speak"],
        "returns": "str",
    },
    "project_development": {
        "module": "actions.dev_agent",
        "function": "dev_agent",
        "params": ["parameters", "response", "player", "session_memory", "speak"],
        "returns": "str",
    },
    "agent_task": {
        "module": "actions.dev_agent",
        "function": "dev_agent",
        "params": ["parameters", "response", "player", "session_memory", "speak"],
        "returns": "str",
    },

    # Computer Control
    "computer_control": {
        "module": "actions.computer_control",
        "function": "computer_control",
        "params": ["parameters", "response", "player", "session_memory"],
        "returns": "str",
    },

    # Screen Capture
    "screen_capture": {
        "module": "actions.screen_processor",
        "function": "screen_process",
        "params": ["parameters", "player", "speak"],
        "returns": "str",
    },

    # Communication
    "send_message": {
        "module": "actions.send_message",
        "function": "send_message",
        "params": ["parameters", "player"],
        "returns": "str",
    },
    "reminder": {
        "module": "actions.reminder",
        "function": "reminder",
        "params": ["parameters", "player"],
        "returns": "str",
    },

    # Games
    "game_updates": {
        "module": "actions.game_updater",
        "function": "game_updater",
        "params": ["parameters", "player", "speak"],
        "returns": "str",
    },

    # Memory
    "memory_save": {
        "module": "actions.memory",
        "function": "memory_action",
        "params": ["parameters", "player"],
        "returns": "str",
    },

    # Generated Code (special case - handled by executor)
    "generated_code": {
        "module": "agent.executor",
        "function": "_execute_generated_code",
        "params": ["parameters", "player", "speak"],
        "returns": "str",
    },
}

# Cache for imported modules
_MODULE_CACHE: Dict[str, Any] = {}


def _get_function(module_name: str, function_name: str) -> Callable:
    """Get a function from a module, caching the module."""
    if module_name not in _MODULE_CACHE:
        _MODULE_CACHE[module_name] = importlib.import_module(module_name)
    module = _MODULE_CACHE[module_name]
    return getattr(module, function_name)


def dispatch_tool(tool_name: str, parameters: Dict = None, player=None, speak=None) -> str:
    """
    Dispatch a tool call by capability tool name.

    Args:
        tool_name: The tool name from capabilities.json (e.g., "open_app", "youtube_play")
        parameters: Dict of parameters from the planner
        player: Optional player/logger object
        speak: Optional TTS callback

    Returns:
        String result from the tool execution

    Raises:
        ValueError: If tool_name is not in dispatch table
        ImportError: If the action module cannot be imported
        Exception: Any exception from the tool itself
    """
    if tool_name not in TOOL_DISPATCH_TABLE:
        raise ValueError(f"Unknown tool: {tool_name}. Available: {list(TOOL_DISPATCH_TABLE.keys())}")

    spec = TOOL_DISPATCH_TABLE[tool_name]

    # Prepare parameters with any injected defaults
    params = dict(parameters or {})
    if "inject" in spec:
        for k, v in spec["inject"].items():
            if k not in params:
                params[k] = v

    # Get the function
    func = _get_function(spec["module"], spec["function"])

    # Build kwargs based on what the function accepts
    import inspect
    sig = inspect.signature(func)
    kwargs = {}

    for param_name in spec["params"]:
        if param_name == "parameters":
            kwargs["parameters"] = params
        elif param_name == "player":
            kwargs["player"] = player
        elif param_name == "speak":
            kwargs["speak"] = speak
        elif param_name == "response":
            kwargs["response"] = None
        elif param_name == "session_memory":
            kwargs["session_memory"] = None

    # Call the function
    result = func(**kwargs)
    return result or "Done."


def list_available_tools() -> list:
    """Return list of all available tool names."""
    return sorted(TOOL_DISPATCH_TABLE.keys())


def get_tool_spec(tool_name: str) -> Dict[str, Any]:
    """Get the dispatch spec for a tool."""
    return TOOL_DISPATCH_TABLE.get(tool_name, {})


if __name__ == "__main__":
    # Test
    print("Available tools:")
    for tool in list_available_tools():
        spec = get_tool_spec(tool)
        print(f"  {tool} -> {spec['module']}.{spec['function']}")
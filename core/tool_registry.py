# core/tool_registry.py
import json
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
TOOLS_PATH = BASE_DIR / "config" / "tools.json"


def _load_tools() -> List[Dict[str, Any]]:
    if not TOOLS_PATH.exists():
        raise FileNotFoundError(f"tools.json not found at {TOOLS_PATH}")
    with TOOLS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    tools = data.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("tools.json 'tools' must be a list")
    return tools


def get_tool_registry() -> List[Dict[str, Any]]:
    return _load_tools()


TOOL_DECLARATIONS = _load_tools()

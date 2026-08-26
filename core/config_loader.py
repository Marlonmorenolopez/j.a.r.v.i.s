"""Central configuration loader for P.I.P.E.
Loads .env file using python-dotenv and provides access to environment variables.
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

def get_base_dir() -> Path:
    """Get the base directory of the project (project root)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    # Go up one level from core/ to project root
    return Path(__file__).resolve().parent.parent

# Load .env file from project root
BASE_DIR = get_base_dir()
ENV_PATH = BASE_DIR / ".env"

# Load environment variables from .env file
# This will not override existing environment variables
load_dotenv(ENV_PATH, override=False)

def get_env(key: str, default: str = "") -> str:
    """Get an environment variable with optional default."""
    return os.environ.get(key, default)

# Configuration constants
GEMINI_API_KEY = get_env("GEMINI_API_KEY")
OPENROUTER_MODEL = get_env("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

# Legacy config file path (for backward compatibility)
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"

def load_legacy_api_keys() -> dict:
    """Load API keys from legacy config file if it exists."""
    if not API_CONFIG_PATH.exists():
        return {}
    try:
        import json
        with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def get_gemini_api_key() -> str:
    """Get Gemini API key from .env or legacy config file."""
    # Priority: .env > legacy config
    if GEMINI_API_KEY:
        return GEMINI_API_KEY
    
    legacy = load_legacy_api_keys()
    return legacy.get("gemini_api_key", "")

def get_openrouter_api_key() -> str:
    """Get OpenRouter API key from .env or legacy config file."""
    key = get_env("OPENROUTER_API_KEY")
    if key:
        return key
    
    legacy = load_legacy_api_keys()
    return legacy.get("openrouter_api_key", "")

def is_configured() -> bool:
    """Check if required API keys are configured."""
    return bool(get_gemini_api_key())

# Export all for convenience
__all__ = [
    "BASE_DIR",
    "ENV_PATH",
    "API_CONFIG_PATH",
    "get_env",
    "GEMINI_API_KEY",
    "OPENROUTER_MODEL",
    "load_legacy_api_keys",
    "get_gemini_api_key",
    "get_openrouter_api_key",
    "is_configured",
]
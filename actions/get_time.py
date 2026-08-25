# actions/get_time.py
# Get current time/date

import sys
from pathlib import Path
from datetime import datetime
import locale

def _get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent

BASE_DIR = _get_base_dir()

# Try to set Spanish locale for formatting
try:
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_TIME, 'Spanish_Spain')
    except:
        pass

def get_time(
    parameters: dict = None,
    player=None,
) -> str:
    """
    Get current time and date.
    Parameters:
        - format: "time" | "date" | "datetime" | "full" (default: "full")
        - timezone: optional timezone string (default: local)
    """
    params = parameters or {}
    fmt = params.get("format", "full").strip().lower()

    now = datetime.now()

    if fmt == "time":
        result = now.strftime("%H:%M:%S")
    elif fmt == "date":
        result = now.strftime("%d de %B de %Y")
    elif fmt == "datetime":
        result = now.strftime("%d de %B de %Y, %H:%M:%S")
    else:  # full
        result = now.strftime("%A, %d de %B de %Y, %H:%M:%S")

    if player:
        player.write_log(f"[Time] {result}")

    return f"Son las {result}"


if __name__ == "__main__":
    print(get_time({}))
    print(get_time({"format": "time"}))
    print(get_time({"format": "date"}))
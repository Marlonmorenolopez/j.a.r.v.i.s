# actions/system_shutdown.py
# System shutdown/restart/logoff for Windows

import sys
import subprocess
from pathlib import Path

def _get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent

BASE_DIR = _get_base_dir()

_DANGEROUS_ACTIONS = {"shutdown", "restart", "logoff", "hibernate", "sleep"}

def system_shutdown(
    parameters: dict = None,
    player=None,
    speak=None,
) -> str:
    """
    Handle system shutdown/restart/logoff.
    Parameters:
        - action: "shutdown" | "restart" | "logoff" | "hibernate" | "sleep"
        - confirmed: "yes" to confirm dangerous actions
        - delay: seconds to wait before action (default 0)
    """
    params = parameters or {}
    action = params.get("action", "").strip().lower()
    confirmed = str(params.get("confirmed", "")).lower()
    delay = int(params.get("delay", 0))

    if action not in _DANGEROUS_ACTIONS:
        return f"Unknown action: {action}. Use: shutdown, restart, logoff, hibernate, sleep"

    # Require confirmation for dangerous actions
    if confirmed not in ("yes", "true", "1", "confirm"):
        return (
            f"This will {action} the computer. "
            f"Please confirm by calling again with confirmed=yes."
        )

    if player:
        player.write_log(f"[System] {action} initiated (delay: {delay}s)")

    try:
        if delay > 0:
            import time
            time.sleep(delay)

        if sys.platform == "win32":
            if action == "shutdown":
                subprocess.run(["shutdown", "/s", "/t", "0"], check=True)
            elif action == "restart":
                subprocess.run(["shutdown", "/r", "/t", "0"], check=True)
            elif action == "logoff":
                subprocess.run(["shutdown", "/l"], check=True)
            elif action == "hibernate":
                subprocess.run(["shutdown", "/h"], check=True)
            elif action == "sleep":
                subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"], check=True)
            return f"System {action} initiated."
        else:
            # Linux/macOS
            if action == "shutdown":
                subprocess.run(["shutdown", "now"], check=True)
            elif action == "restart":
                subprocess.run(["reboot"], check=True)
            elif action == "logoff":
                subprocess.run(["pkill", "-u", "$USER"], check=True)
            elif action == "hibernate":
                subprocess.run(["systemctl", "hibernate"], check=True)
            elif action == "sleep":
                subprocess.run(["systemctl", "suspend"], check=True)
            return f"System {action} initiated."

    except subprocess.CalledProcessError as e:
        return f"Failed to {action}: {e}"
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    # Test (won't actually shutdown without confirmed=yes)
    print(system_shutdown({"action": "shutdown", "confirmed": "no"}))
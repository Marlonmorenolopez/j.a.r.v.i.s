# actions/memory.py
# Memory save/recall action

import sys
from pathlib import Path

def _get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent

BASE_DIR = _get_base_dir()
sys.path.insert(0, str(BASE_DIR))

from memory.memory_manager import load_memory, save_memory, update_memory, remember, forget

def memory_save(
    parameters: dict = None,
    player=None,
) -> str:
    """
    Save content to long-term memory.
    Parameters:
        - content: string to save (required)
        - category: memory category (default: "general")
        - key: optional key name (default: auto-generated timestamp)
    """
    params = parameters or {}
    content = params.get("content", "").strip()
    category = params.get("category", "general").strip()
    key = params.get("key", "").strip()

    if not content:
        return "No content provided to save."

    if not key:
        from datetime import datetime
        key = f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        result = remember(key, content, category)

        if player:
            player.write_log(f"[Memory] Saved to {category}/{key}")

        return f"Guardado en memoria ({category}/{key})."

    except Exception as e:
        return f"Error guardando en memoria: {e}"


def memory_recall(
    parameters: dict = None,
    player=None,
) -> str:
    """
    Recall content from long-term memory.
    Parameters:
        - query: search query (optional, returns all if empty)
        - category: specific category to search (optional)
        - limit: max results (default: 10)
    """
    params = parameters or {}
    query = params.get("query", "").strip().lower()
    category = params.get("category", "").strip()
    limit = int(params.get("limit", 10))

    try:
        memory = load_memory()

        results = []
        categories_to_search = [category] if category else list(memory.keys())

        for cat in categories_to_search:
            if cat not in memory:
                continue
            for key, entry in memory[cat].items():
                if isinstance(entry, dict) and "value" in entry:
                    value = entry["value"]
                    if not query or query in value.lower() or query in key.lower():
                        results.append(f"[{cat}/{key}] {value}")
                        if len(results) >= limit:
                            break

        if not results:
            return "No se encontraron entradas en memoria."

        return "Memoria:\n" + "\n".join(results)

    except Exception as e:
        return f"Error leyendo memoria: {e}"


def memory_action(
    parameters: dict = None,
    player=None,
) -> str:
    """
    Unified memory action - routes to save or recall based on parameters.
    """
    params = parameters or {}
    action = params.get("action", "").strip().lower()

    if action in ("save", "guardar", "store"):
        return memory_save(parameters, player)
    elif action in ("recall", "recordar", "get", "read", "leer"):
        return memory_recall(parameters, player)
    else:
        # Default: if content provided -> save, else -> recall
        if params.get("content"):
            return memory_save(parameters, player)
        return memory_recall(parameters, player)


if __name__ == "__main__":
    # Test
    print(memory_action({"action": "save", "content": "Test memory entry", "category": "test"}))
    print(memory_action({"action": "recall", "query": "test"}))
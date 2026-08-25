from core.config_loader import get_gemini_api_key, get_openrouter_api_key, get_base_dir

BASE_DIR = get_base_dir()
CONFIG_DIR  = BASE_DIR / "config"
CONFIG_FILE = CONFIG_DIR / "api_keys.json"


def ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def config_exists() -> bool:
    return CONFIG_FILE.exists()


def save_api_keys(gemini_api_key: str) -> None:
    ensure_config_dir()

    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}

    data["gemini_api_key"] = gemini_api_key.strip()

    CONFIG_FILE.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )


def load_api_keys() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ Failed to load api_keys.json: {e}")
        return {}


def get_gemini_key() -> str | None:
    # Priority: .env > legacy config
    key = get_gemini_api_key()
    if key:
        return key
    return load_api_keys().get("gemini_api_key")


def get_openrouter_key() -> str | None:
    # Priority: .env > legacy config
    key = get_openrouter_api_key()
    if key:
        return key
    return load_api_keys().get("openrouter_api_key")


def is_configured() -> bool:
    # Check both keys are configured
    gemini_key = get_gemini_key()
    openrouter_key = get_openrouter_key()
    return bool(gemini_key and len(gemini_key) > 15 and openrouter_key)
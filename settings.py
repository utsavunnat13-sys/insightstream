import os
import json
from pydantic import BaseModel
from typing import Optional
import dotenv

# Load environment variables
dotenv.load_dotenv()

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

class AppSettings(BaseModel):
    provider: str = "local"  # gemini, openai, anthropic, ollama, local
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = "gemini-3.5-flash"
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = "gpt-4o-mini"
    anthropic_api_key: Optional[str] = None
    anthropic_model: Optional[str] = "claude-3-5-sonnet-latest"
    ollama_api_base: Optional[str] = "http://localhost:11434/v1"
    ollama_model: Optional[str] = "llama3"

# In-memory active settings instance
_active_settings: Optional[AppSettings] = None

def get_default_settings() -> AppSettings:
    """Read initial defaults from environment variables."""
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    use_local_engine = os.environ.get("USE_LOCAL_ENGINE", "false").lower() == "true"
    
    # Decide default provider
    if use_local_engine:
        provider = "local"
    elif gemini_key:
        provider = "gemini"
    elif openai_key:
        provider = "openai"
    elif anthropic_key:
        provider = "anthropic"
    else:
        provider = "local"
        
    return AppSettings(
        provider=provider,
        gemini_api_key=gemini_key,
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
        openai_api_key=openai_key,
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        anthropic_api_key=anthropic_key,
        anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        ollama_api_base=os.environ.get("OLLAMA_API_BASE", "http://localhost:11434/v1"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "llama3")
    )

def load_settings() -> AppSettings:
    """Loads settings from settings.json or defaults from environment variables."""
    global _active_settings
    if _active_settings is not None:
        return _active_settings
        
    defaults = get_default_settings()
    
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)
            # Use defaults as fallback for missing fields in JSON
            merged = defaults.model_dump()
            merged.update(data)
            _active_settings = AppSettings(**merged)
        except Exception as e:
            print(f"Error loading settings.json: {e}")
            _active_settings = defaults
    else:
        _active_settings = defaults
        
    return _active_settings

def save_settings(settings: AppSettings) -> None:
    """Saves settings to settings.json and syncs environment variables in-memory."""
    global _active_settings
    _active_settings = settings
    try:
        with open(SETTINGS_FILE, "w") as f:
            f.write(settings.model_dump_json(indent=2))
        
        # Sync back to environment variables in memory
        if settings.gemini_api_key:
            os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
        if settings.openai_api_key:
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        if settings.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
    except Exception as e:
        print(f"Error saving settings: {e}")
        raise e

def mask_key(key: Optional[str]) -> Optional[str]:
    """Helper to mask sensitive keys for display in UI."""
    if not key:
        return None
    if len(key) <= 8:
        return "...***"
    return f"{key[:4]}...{key[-4:]}"

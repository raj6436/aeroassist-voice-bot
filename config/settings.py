"""
Application settings loaded from environment variables.

Uses python-dotenv to load a .env file at import time so that
every other module can simply ``from config.settings import settings``.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env from project root (two levels up from config/)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


class Settings(BaseModel):
    """Typed application settings sourced from environment variables."""

    # LiveKit
    livekit_url: str = Field(default_factory=lambda: os.getenv("LIVEKIT_URL", ""))
    livekit_api_key: str = Field(default_factory=lambda: os.getenv("LIVEKIT_API_KEY", ""))
    livekit_api_secret: str = Field(default_factory=lambda: os.getenv("LIVEKIT_API_SECRET", ""))

    # Deepgram (STT)
    deepgram_api_key: str = Field(default_factory=lambda: os.getenv("DEEPGRAM_API_KEY", ""))

    # Google Gemini (LLM)
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    # Cartesia (TTS)
    cartesia_api_key: str = Field(default_factory=lambda: os.getenv("CARTESIA_API_KEY", ""))


# Singleton instance — import this everywhere
settings = Settings()


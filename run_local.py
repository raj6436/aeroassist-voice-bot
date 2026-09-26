"""
Local runner script for AeroAssist LiveKit Voice Agent.
Loads configuration and boots the LiveKit Agent worker CLI.
"""

import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure root directory and load environment variables
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aeroassist.runner")


def print_banner():
    banner = """
=====================================================
            === AeroAssist Voice Agent ===
=====================================================
Make sure .env has valid API keys:
  - LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
  - DEEPGRAM_API_KEY, GEMINI_API_KEY, CARTESIA_API_KEY

Run with:
  python run_local.py dev
  python run_local.py start
=====================================================
"""
    print(banner)


if __name__ == "__main__":
    print_banner()

    # Import agent entrypoint and LiveKit CLI
    from agent.core_agent import entrypoint
    from livekit.agents import WorkerOptions, cli

    # Start the worker CLI
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


"""
Core LiveKit Voice Agent entrypoint for AeroAssist.
Connects real-time Audio I/O, Silero VAD, Deepgram STT, Gemini LLM, Cartesia TTS, and business tools.
"""

import os
import sys
import logging

# Ensure shims from agent/__init__.py are loaded
if "livekit.agents.pipeline" not in sys.modules:
    import agent  # noqa: F401

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import silero, deepgram, cartesia, google
from config.prompts import AEROASSIST_SYSTEM_PROMPT
from agent.tools import FlightTools
from agent.state_manager import ConversationState

logger = logging.getLogger("aeroassist")


async def entrypoint(ctx: JobContext):
    """
    LiveKit Agents room entrypoint.
    Subscribes to incoming audio and orchestrates conversational pipeline.
    """
    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Initialize conversation state and tools
    state = ConversationState()
    fnc_ctx = FlightTools(state=state)

    # Initial system prompt
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=AEROASSIST_SYSTEM_PROMPT,
    )

    # Set up the Voice Pipeline Agent
    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(language="hi"),
        llm=google.LLM(model="gemini-2.0-flash"),
        tts=cartesia.TTS(),
        chat_ctx=initial_ctx,
        fnc_ctx=fnc_ctx,
    )

    agent.start(ctx.room)

    # Agent greets first
    await agent.say(
        "Namaste! AeroAssist flight customer support mein aapka swagat hai. Main aapki kya madad kar sakti hoon? Kripya apna PNR number bataiye.",
        allow_interruptions=True,
    )


def main():
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == "__main__":
    main()


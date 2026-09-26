"""
Core LiveKit Voice Agent entrypoint for AeroAssist.
Wires real-time Audio I/O, Silero VAD, Deepgram STT, Gemini LLM, Cartesia TTS,
and business tools together using the real LiveKit Agents 1.x
(`livekit-agents==1.8.2`) Agent / AgentSession API.
"""

import logging

from livekit.agents import Agent, AgentSession, AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.plugins import cartesia, deepgram, google, silero

from agent.state_manager import ConversationState
from agent.tools import FLIGHT_TOOLS
from config.prompts import AEROASSIST_SYSTEM_PROMPT
from config.settings import settings

logger = logging.getLogger("aeroassist")


class AeroAssistAgent(Agent):
    """AeroAssist flight-support voice persona with its business tools attached."""

    def __init__(self) -> None:
        super().__init__(instructions=AEROASSIST_SYSTEM_PROMPT, tools=FLIGHT_TOOLS)


async def entrypoint(ctx: JobContext):
    """
    LiveKit Agents room entrypoint.
    Connects to the room and starts the voice pipeline (VAD -> STT -> LLM -> tools -> TTS).
    """
    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Per-call conversation state, shared with the tools via AgentSession.userdata
    state = ConversationState()

    # api_key is passed explicitly because livekit-plugins-google reads
    # GOOGLE_API_KEY by default, while this project's .env / config.settings
    # use GEMINI_API_KEY.
    session = AgentSession[ConversationState](
        vad=silero.VAD.load(),
        stt=deepgram.STT(language="hi", api_key=settings.deepgram_api_key),
        llm=google.LLM(model="gemini-3.8-flash", api_key=settings.gemini_api_key),
        tts=cartesia.TTS(api_key=settings.cartesia_api_key),
        userdata=state,
    )

    await session.start(agent=AeroAssistAgent(), room=ctx.room)

    # Agent greets first
    await session.say(
        "Namaste! AeroAssist flight customer support mein aapka swagat hai. "
        "Main aapki kya madad kar sakti hoon? Kripya apna PNR number bataiye.",
        allow_interruptions=True,
    )


def main():
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == "__main__":
    main()

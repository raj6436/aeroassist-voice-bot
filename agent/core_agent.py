"""
Core LiveKit Voice Agent entrypoint for AeroAssist.
Wires real-time Audio I/O, Silero VAD, Deepgram STT, Gemini LLM, Cartesia TTS,
and business tools together using the real LiveKit Agents 1.x
(`livekit-agents==1.8.2`) Agent / AgentSession API.
"""

import asyncio
import logging

from livekit.agents import Agent, AgentSession, AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.voice.events import ErrorEvent, LLMError
from livekit.plugins import cartesia, deepgram, google, silero

from agent.escalation_handler import EscalationHandler
from agent.state_manager import ConversationState
from agent.tools import FLIGHT_TOOLS
from config.prompts import AEROASSIST_SYSTEM_PROMPT
from config.settings import settings

# Spoken once, the moment the LLM permanently fails (retries exhausted) instead
# of the agent going silent. Kept short and in the same bilingual voice-agent
# style as AEROASSIST_SYSTEM_PROMPT.
LLM_FAILURE_FALLBACK_LINE = "Mujhe thoda technical issue aa raha hai, ek second dijiye."

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
        # Explicit turn/interruption handling for a phone-style conversation:
        # barge-in is allowed, but a brief filler sound/cough (<0.5s) or an
        # empty STT result won't be treated as a real interruption.
        turn_handling={
            "turn_detection": "vad",
            "interruption": {
                "enabled": True,
                "min_duration": 0.5,
                "min_words": 0,
            },
            "endpointing": {
                "min_delay": 0.5,
                "max_delay": 3.0,
            },
        },
    )

    def _on_llm_permanent_failure(ev: ErrorEvent) -> None:
        """
        The LLM (Gemini) can fail permanently after LiveKit's own retries are
        exhausted (e.g. repeated 503s/timeouts). Left unhandled, the agent
        just goes silent forever. Speak a short apology, then route this
        exactly like any other unresolved-issue escalation: reuse
        EscalationHandler's hold message and ConversationState's existing
        handoff-briefing text (same content a human agent would get from the
        escalate_to_human tool) — the bot genuinely cannot help once the LLM
        itself is down.
        """
        if not isinstance(ev.error, LLMError) or ev.error.recoverable:
            return
        if state.escalation_triggered:
            return  # already escalating; avoid repeating on further failures

        state.escalation_triggered = True
        state.escalation_reason = "LLM permanently unavailable (technical failure after retries)"
        state.record_tool("llm_failure_escalation", state.escalation_reason)

        # Spoken as ONE session.say() call, not two: two sequential calls each
        # incur their own separate TTS round-trip, and the scheduler plays
        # queued speech one at a time — that produced a ~5s silent gap between
        # the fallback line and the hold message (confirmed via diagnostic
        # instrumentation), which sounded like the agent had stopped.
        speech_handle = session.say(
            f"{LLM_FAILURE_FALLBACK_LINE} {EscalationHandler.generate_hold_message()}",
            allow_interruptions=True,
        )
        logger.warning(
            "LLM permanently failed; escalating to human.\n%s",
            state.get_summary_for_handoff(),
        )

        # TEMPORARY DIAGNOSTIC INSTRUMENTATION — remove once the fix is
        # confirmed. Proves what actually happens to the combined SpeechHandle
        # (played out vs. cancelled vs. errored) instead of guessing.
        async def _log_speech_outcome(name: str, handle) -> None:
            await handle  # never raises; failure surfaces via .exception()
            logger.warning(
                "[DIAG] %s SpeechHandle outcome: done=%s interrupted=%s scheduled=%s exception=%r",
                name,
                handle.done(),
                handle.interrupted,
                handle.scheduled,
                handle.exception(),
            )

        asyncio.create_task(_log_speech_outcome("fallback_plus_hold", speech_handle))

    session.on("error", _on_llm_permanent_failure)

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

"""
Core LiveKit Voice Agent entrypoint for AeroAssist.
Wires real-time Audio I/O, Silero VAD, Deepgram STT, Gemini LLM, Cartesia TTS,
and business tools together using the real LiveKit Agents 1.x
(`livekit-agents==1.8.2`) Agent / AgentSession API.
"""

import asyncio
import logging

from livekit.agents import Agent, AgentSession, AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.voice.events import (
    AgentStateChangedEvent,
    CloseEvent,
    ConversationItemAddedEvent,
    ErrorEvent,
    EotPredictionEvent,
    FunctionToolsExecutedEvent,
    LLMError,
    ToolExecutionUpdatedEvent,
    UserInputTranscribedEvent,
)
from livekit.plugins import cartesia, deepgram, google, silero

from agent.escalation_handler import EscalationHandler
from agent.event_log import log_event
from agent.state_manager import ConversationState
from agent.tools import FLIGHT_TOOLS
from config.prompts import AEROASSIST_SYSTEM_PROMPT
from config.settings import settings

# Spoken once, the moment the LLM permanently fails (retries exhausted) instead
# of the agent going silent. Kept short and in the same bilingual voice-agent
# style as AEROASSIST_SYSTEM_PROMPT.
LLM_FAILURE_FALLBACK_LINE = "Mujhe thoda technical issue aa raha hai, ek second dijiye."

# Spoken on every permanent LLM failure AFTER the first one in the same call.
# Without this, a second/third permanent failure went completely silent: the
# full escalation/hold message is only ever spoken once (state.escalation_triggered
# guards that), but the call can keep failing after that - confirmed via a real
# Playground session where Gemini returned repeated 503/504/429s and the agent
# produced no audio at all for the user's last two turns before they hung up.
LLM_FAILURE_REPEAT_FALLBACK_LINE = "Abhi bhi technical issue aa raha hai. Kripya thodi der line par rahiye."

logger = logging.getLogger("aeroassist")
diag_logger = logging.getLogger("aeroassist.latency_diag")


# TEMPORARY DIAGNOSTIC INSTRUMENTATION — latency investigation only, remove
# once the investigation's optimization plan has been reviewed and applied.
# Logs the LiveKit Agents SDK's own native per-turn timing (ChatMessage.metrics,
# EotPredictionEvent, ToolExecutionUpdatedEvent) rather than adding custom
# timers, per the investigation's instruction to prefer SDK-native events.
# Never logs secrets; transcript text is logged (same as the SDK's own debug
# logs already do) purely to correlate turns with the benchmark utterances.
def _install_latency_diagnostics(session: AgentSession) -> None:
    _tool_call_started_at: dict[str, float] = {}

    def _on_user_input_transcribed(ev: UserInputTranscribedEvent) -> None:
        diag_logger.info(
            "[LATDIAG] user_input_transcribed is_final=%s item_id=%s transcript=%r t=%.3f",
            ev.is_final, ev.item_id, ev.transcript, ev.created_at,
        )

    def _on_eot_prediction(ev: EotPredictionEvent) -> None:
        diag_logger.info(
            "[LATDIAG] eot_prediction probability=%.3f threshold=%.3f "
            "inference_duration=%.3f delay=%.3f t=%.3f",
            ev.probability, ev.threshold, ev.inference_duration, ev.delay, ev.created_at,
        )

    def _on_conversation_item_added(ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if getattr(item, "type", None) != "message":
            return
        diag_logger.info(
            "[LATDIAG] conversation_item_added role=%s text=%r metrics=%s t=%.3f",
            item.role, item.text_content, dict(item.metrics), ev.created_at,
        )

    def _on_tool_execution_updated(ev: ToolExecutionUpdatedEvent) -> None:
        update = ev.update
        if update.type == "tool_call_started":
            _tool_call_started_at[update.function_call.call_id] = ev.created_at
            diag_logger.info(
                "[LATDIAG] tool_call_started name=%s call_id=%s t=%.3f",
                update.function_call.name, update.function_call.call_id, ev.created_at,
            )
        elif update.type == "tool_call_ended":
            started_at = _tool_call_started_at.pop(update.call_id, None)
            duration = (ev.created_at - started_at) if started_at is not None else None
            diag_logger.info(
                "[LATDIAG] tool_call_ended call_id=%s status=%s duration=%s t=%.3f",
                update.call_id, update.status,
                f"{duration:.3f}" if duration is not None else "unknown",
                ev.created_at,
            )

    session.on("user_input_transcribed", _on_user_input_transcribed)
    session.on("eot_prediction", _on_eot_prediction)
    session.on("conversation_item_added", _on_conversation_item_added)
    session.on("tool_execution_updated", _on_tool_execution_updated)


# Structured call/event logging for the operational dashboard (agent/event_log.py).
# Read-only from the voice pipeline's perspective: every hook here only
# observes events the AgentSession already emits and never alters session
# behavior, state, or control flow. log_event() itself never blocks the event
# loop (see agent/event_log.py), so this cannot add latency to the voice path.
def _install_event_logging(session: AgentSession, call: str, state: ConversationState) -> None:
    def _on_user_input_transcribed(ev: UserInputTranscribedEvent) -> None:
        if ev.is_final:
            log_event(call, "user_speech", input=ev.transcript, language=ev.language, stage="STT (Deepgram)")

    def _on_conversation_item_added(ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if getattr(item, "type", None) == "message" and item.role == "assistant":
            log_event(call, "agent_reply", output=item.text_content, stage="LLM (Gemini) -> TTS (Cartesia)")

    def _on_tools(ev: FunctionToolsExecutedEvent) -> None:
        for fc, out in zip(ev.function_calls, ev.function_call_outputs, strict=False):
            log_event(
                call, "tool_call", source=f"agent/tools.py:{fc.name}",
                input=fc.arguments, output=getattr(out, "output", None) if out else None,
                is_error=bool(getattr(out, "is_error", False)) or None,
            )

    def _on_state(ev: AgentStateChangedEvent) -> None:
        log_event(call, "agent_state", output=f"{ev.old_state} -> {ev.new_state}")

    def _on_error(ev: ErrorEvent) -> None:
        log_event(call, "error", output=f"{type(ev.error).__name__}: {ev.error}"[:500], stage=str(ev.source)[:80])

    def _on_close(ev: CloseEvent) -> None:
        log_event(
            call, "call_end", output=f"reason={getattr(ev, 'reason', '')}",
            escalated=state.escalation_triggered or None, pnr=state.verified_pnr,
            passenger=state.passenger_name,
        )

    session.on("user_input_transcribed", _on_user_input_transcribed)
    session.on("conversation_item_added", _on_conversation_item_added)
    session.on("function_tools_executed", _on_tools)
    session.on("agent_state_changed", _on_state)
    session.on("error", _on_error)
    session.on("close", _on_close)


def _on_llm_permanent_failure(ev: ErrorEvent, session: AgentSession, state: ConversationState) -> None:
    """
    The LLM (Gemini) can fail permanently after LiveKit's own retries are
    exhausted (e.g. repeated 503s/timeouts). Left unhandled, the agent just
    goes silent forever.

    FIRST permanent failure in a call: speak a short apology plus the existing
    hold/escalation message (same content a human agent would get from the
    escalate_to_human tool), and latch escalation_triggered - the bot
    genuinely cannot help once the LLM itself is down.

    Any SUBSEQUENT permanent failure in the SAME call: the full escalation/
    handoff message must not repeat (same reasoning, no new information), but
    staying silent is wrong too - the call can keep failing after the first
    escalation, and the user otherwise hears nothing at all. Speak a short,
    repeatable fallback line instead.
    """
    if not isinstance(ev.error, LLMError) or ev.error.recoverable:
        return

    if state.escalation_triggered:
        session.say(LLM_FAILURE_REPEAT_FALLBACK_LINE, allow_interruptions=True)
        return

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

    call = ctx.room.name
    log_event(call, "call_start", direction="inbound", output="agent joined room")

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
    _install_latency_diagnostics(session)
    _install_event_logging(session, call, state)

    session.on("error", lambda ev: _on_llm_permanent_failure(ev, session, state))

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

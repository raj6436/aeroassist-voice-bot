"""
Core LiveKit Voice Agent entrypoint for AeroAssist.
Wires real-time Audio I/O, Silero VAD, Deepgram STT, Gemini LLM, Cartesia TTS,
and business tools together using the real LiveKit Agents 1.x
(`livekit-agents==1.8.2`) Agent / AgentSession API.
"""

import array
import asyncio
import logging

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    APIStatusError,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.agents.metrics import LLMMetrics
from livekit.agents.voice.events import (
    AgentStateChangedEvent,
    CloseEvent,
    ConversationItemAddedEvent,
    ErrorEvent,
    EotPredictionEvent,
    FunctionToolsExecutedEvent,
    LLMError,
    MetricsCollectedEvent,
    ToolExecutionUpdatedEvent,
    UserInputTranscribedEvent,
)
from google.genai import types as genai_types
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


# TEMPORARY DIAGNOSTIC INSTRUMENTATION — inbound-caller-audio investigation
# only, remove once that investigation concludes. Uses the raw rtc.Room
# events the LiveKit RTC SDK already fires (participant_connected,
# track_published, track_subscribed, track_subscription_failed) rather than
# anything AgentSession-specific, so this observes the exact same thing for
# both the Twilio-Connector phone path and the existing browser/WebRTC path
# with zero behavior difference between them - purely additive logging, no
# control-flow change, no effect on VAD/STT/LLM/TTS.
def _install_room_track_diagnostics(room) -> None:
    def _on_participant_connected(participant) -> None:
        diag_logger.info(
            "[LATDIAG] participant_connected identity=%s kind=%s",
            participant.identity, participant.kind,
        )

    def _on_track_published(publication, participant) -> None:
        diag_logger.info(
            "[LATDIAG] track_published participant=%s sid=%s kind=%s source=%s muted=%s",
            participant.identity, publication.sid, publication.kind, publication.source, publication.muted,
        )

    def _on_track_subscribed(track, publication, participant) -> None:
        diag_logger.info(
            "[LATDIAG] track_subscribed participant=%s sid=%s kind=%s source=%s",
            participant.identity, publication.sid, track.kind, publication.source,
        )
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(_observe_audio_frames(track, participant.identity))

    def _on_track_subscription_failed(participant, track_sid, error) -> None:
        diag_logger.warning(
            "[LATDIAG] track_subscription_failed participant=%s sid=%s error=%s",
            participant.identity, track_sid, error,
        )

    room.on("participant_connected", _on_participant_connected)
    room.on("track_published", _on_track_published)
    room.on("track_subscribed", _on_track_subscribed)
    room.on("track_subscription_failed", _on_track_subscription_failed)


# TEMPORARY DIAGNOSTIC INSTRUMENTATION — inbound-caller-audio investigation
# only, remove once that investigation concludes. Opens a SECOND, independent
# rtc.AudioStream on the already-subscribed track purely to observe it.
# Confirmed safe by reading the installed SDK's own audio_stream.py: each
# AudioStream opens its own "owned" native stream keyed by track_handle, and
# LiveKit's native layer fans frames out to every AudioStream registered on a
# track - this is the same mechanism AgentSession's own separate AudioStream
# uses to read this track, so this neither steals, delays, nor alters the
# frames AgentSession/VAD/Deepgram actually receive. Read-only: no frame is
# ever written back, forwarded, resampled in place, or muted - only inspected
# for sample_rate/channels/amplitude, then discarded. Logs periodic summaries
# (not per-frame) to avoid flooding the log.
async def _observe_audio_frames(track: rtc.Track, participant_identity: str) -> None:
    audio_stream = rtc.AudioStream.from_track(track=track)
    frame_count = 0
    nonzero_frame_count = 0
    window_frames = 0
    window_max_amplitude = 0
    logged_format = False
    try:
        async for event in audio_stream:
            frame = event.frame
            frame_count += 1
            window_frames += 1

            if not logged_format:
                logged_format = True
                frame_duration_ms = (frame.samples_per_channel / frame.sample_rate) * 1000
                diag_logger.info(
                    "[LATDIAG] audio_frame_format participant=%s sample_rate=%d num_channels=%d "
                    "samples_per_channel=%d frame_duration_ms=%.2f",
                    participant_identity, frame.sample_rate, frame.num_channels,
                    frame.samples_per_channel, frame_duration_ms,
                )

            samples = array.array("h")
            samples.frombytes(bytes(frame.data))
            frame_max_amplitude = max((abs(s) for s in samples), default=0)
            window_max_amplitude = max(window_max_amplitude, frame_max_amplitude)
            # -32768..32767 is the full int16 range; 50 is a small noise-floor
            # threshold well below any real speech, chosen only to separate
            # "silence/line noise" from "something with real signal arrived".
            if frame_max_amplitude > 50:
                nonzero_frame_count += 1

            if window_frames >= 50:
                diag_logger.info(
                    "[LATDIAG] audio_frame_window participant=%s frames_total=%d "
                    "nonzero_frames_total=%d window_max_amplitude=%d",
                    participant_identity, frame_count, nonzero_frame_count, window_max_amplitude,
                )
                window_frames = 0
                window_max_amplitude = 0
    except Exception as e:
        diag_logger.warning(
            "[LATDIAG] audio_frame_observation_error participant=%s frames_total=%d error=%r",
            participant_identity, frame_count, e,
        )
    finally:
        diag_logger.info(
            "[LATDIAG] audio_frame_stream_closed participant=%s frames_total=%d nonzero_frames_total=%d",
            participant_identity, frame_count, nonzero_frame_count,
        )
        await audio_stream.aclose()


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


def _extract_llm_usage(metrics: LLMMetrics) -> dict:
    """
    Pull the structured usage/timing fields out of a native LLMMetrics event
    into the plain dict shape used for diagnostic logging. Defensive against
    any field being absent on a given SDK/provider combination - every value
    falls back to None rather than raising, so this never breaks logging.
    """
    return {
        "event": "llm_metrics",
        "prompt_tokens": getattr(metrics, "prompt_tokens", None),
        "completion_tokens": getattr(metrics, "completion_tokens", None),
        "total_tokens": getattr(metrics, "total_tokens", None),
        "ttft": getattr(metrics, "ttft", None),
        "tokens_per_second": getattr(metrics, "tokens_per_second", None),
        # Added alongside the thinking_level=LOW change above: reasoning_tokens
        # is already counted inside completion_tokens, but surfacing it
        # separately is the only way to observe from logs whether capping the
        # thinking level actually reduced hidden "thinking" token spend.
        "reasoning_tokens": getattr(metrics, "reasoning_tokens", None),
    }


# TEMPORARY DIAGNOSTIC INSTRUMENTATION — latency/token-usage investigation
# only. ``metrics_collected`` is deprecated by the installed SDK (1.8.3) in
# favor of ``session_usage_updated`` (session-level cumulative totals only,
# no per-request prompt/completion token breakdown or TTFT) and
# ``ChatMessage.metrics`` (per-turn timing, but its MetricsReport TypedDict
# has no token-count fields at all - confirmed by reading both event classes'
# installed source). ``metrics_collected`` is still the ONLY exposed source
# of real per-request Gemini token counts in this version, and it is still
# actively emitted (agent_activity.py forwards llm/stt/tts/vad metrics into
# it) - only registering a handler prints one deprecation warning, nothing
# is actually removed. Used here deliberately, per that constraint, rather
# than inventing an unsupported API. Never logs secrets: LLMMetrics carries
# token counts, timing, and a provider request_id, never prompt/response
# text or API keys.
def _install_llm_usage_metrics(session: AgentSession) -> None:
    def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        if ev.metrics.type != "llm_metrics":
            return
        diag_logger.info("[LLM_METRICS] %s", _extract_llm_usage(ev.metrics))

    session.on("metrics_collected", _on_metrics_collected)


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


def _is_quota_exceeded_error(error: LLMError) -> bool:
    """
    True if this LLMError's underlying exception is a Gemini HTTP 429
    (RESOURCE_EXHAUSTED / quota or rate-limit). ``status_code`` is a public,
    documented attribute of ``APIStatusError`` (livekit-agents==1.8.3) -
    confirmed by reading the installed SDK source, not guessed.

    NOTE on what this does NOT do: LiveKit's own internal LLM retry loop
    (``LLMStream._main_task`` in livekit-agents core) decides whether to
    retry purely from a per-exception ``retryable`` boolean, with no
    per-error-type hook we can intercept - the Google plugin marks 429 as
    ``retryable=True`` (same as a transient 503/504), so a 429 is retried
    the same number of times (default ``max_retry=3``) before LiveKit
    itself emits ``recoverable=False``. There is no SDK-supported way to
    shortcut that wait for 429 specifically without either globally
    lowering ``max_retry`` (which would also cut retries for genuinely
    transient 503/504 errors - not done here) or speaking before the SDK
    has given up (which risks a second, contradictory reply if a later
    retry in the same burst succeeds - not done here either). This
    function only classifies the failure once LiveKit has already decided
    it is permanent, so the existing escalation timing is unchanged.
    """
    return isinstance(error.error, APIStatusError) and error.error.status_code == 429


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

    quota_exceeded = _is_quota_exceeded_error(ev.error)
    diag_logger.warning(
        "[LATDIAG] llm_permanent_failure quota_exceeded=%s status_code=%s already_escalated=%s",
        quota_exceeded,
        getattr(ev.error.error, "status_code", None),
        state.escalation_triggered,
    )

    if state.escalation_triggered:
        session.say(LLM_FAILURE_REPEAT_FALLBACK_LINE, allow_interruptions=True)
        return

    state.escalation_triggered = True
    state.escalation_reason = (
        "LLM permanently unavailable (Gemini quota/rate-limit exceeded - HTTP 429)"
        if quota_exceeded
        else "LLM permanently unavailable (technical failure after retries)"
    )
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
    _install_room_track_diagnostics(ctx.room)

    call = ctx.room.name
    log_event(call, "call_start", direction="inbound", output="agent joined room")

    # Per-call conversation state, shared with the tools via AgentSession.userdata
    state = ConversationState()

    # api_key is passed explicitly because livekit-plugins-google reads
    # GOOGLE_API_KEY by default, while this project's .env / config.settings
    # use GEMINI_API_KEY.
    #
    # thinking_level=LOW: gemini-3.8-flash defaults to an automatic (model-
    # decided) thinking budget when this is left unset. Real-call measurements
    # showed Gemini TTFT varying 1.6s-8.5s with NO correlation to prompt size
    # (the smallest prompt of the call had the largest TTFT) - consistent with
    # variable server-side "thinking" time, not prompt processing. Our tasks
    # (PNR/flight-status/refund lookups via 5 clearly-named tools, short
    # scripted replies) don't need deep multi-step reasoning, so capping
    # thinking to LOW is a low-risk, one-line, fully reversible lever aimed
    # directly at that TTFT variance. Needs a real call to confirm the actual
    # effect - see the [LLM_METRICS] reasoning_tokens field added below.
    session = AgentSession[ConversationState](
        vad=silero.VAD.load(),
        stt=deepgram.STT(language="hi", api_key=settings.deepgram_api_key),
        llm=google.LLM(
            model="gemini-3.8-flash",
            api_key=settings.gemini_api_key,
            thinking_config=genai_types.ThinkingConfig(thinking_level=genai_types.ThinkingLevel.LOW),
        ),
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
    _install_llm_usage_metrics(session)
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

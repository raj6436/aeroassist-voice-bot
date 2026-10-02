"""
Unit tests for agent/core_agent.py's _on_llm_permanent_failure.

Covers the fix for a real, diagnosed bug: a second (or later) permanent LLM
failure in the same call used to produce total silence, because the full
escalation/hold message is only meant to be spoken once. These tests prove:
first permanent failure -> full escalation message (unchanged behavior),
every later permanent failure -> a short, repeatable fallback instead of
silence, and recoverable/non-LLM errors are still ignored exactly as before.

AgentSession is mocked (no real LiveKit/TTS/network involved) - only our own
_on_llm_permanent_failure logic and the real ConversationState are exercised.

_on_llm_permanent_failure fires asyncio.create_task() on the first-failure
path (pre-existing diagnostic instrumentation, unchanged by this fix), which
needs a running event loop - _fire() below runs each call inside one.
"""

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from livekit.agents.voice.events import ErrorEvent, LLMError

from agent.core_agent import (
    LLM_FAILURE_FALLBACK_LINE,
    LLM_FAILURE_REPEAT_FALLBACK_LINE,
    _on_llm_permanent_failure,
)
from agent.escalation_handler import EscalationHandler
from agent.state_manager import ConversationState


class _FakeSpeechHandle:
    """Minimal awaitable stand-in for LiveKit's SpeechHandle."""

    def __await__(self):
        return iter([])

    def done(self):
        return True

    interrupted = False
    scheduled = True

    def exception(self):
        return None


def _fire(ev: ErrorEvent, session, state) -> None:
    async def _run():
        _on_llm_permanent_failure(ev, session, state)
        await asyncio.sleep(0)  # let the fire-and-forget diagnostic task finish

    asyncio.run(_run())


def _llm_error_event(recoverable: bool) -> ErrorEvent:
    error = LLMError(timestamp=0.0, label="llm", error=Exception("boom"), recoverable=recoverable)
    return ErrorEvent(error=error, source=object())


def _non_llm_error_event() -> ErrorEvent:
    class _OtherError:
        recoverable = False

    return ErrorEvent(error=_OtherError(), source=object())


def _make_session() -> MagicMock:
    session = MagicMock()
    session.say = MagicMock(return_value=_FakeSpeechHandle())
    return session


class TestFirstPermanentFailure(unittest.TestCase):
    def setUp(self):
        self.session = _make_session()
        self.state = ConversationState()

    def test_speaks_full_fallback_and_hold_message(self):
        with patch.object(EscalationHandler, "generate_hold_message", return_value="HOLD_MSG"):
            _fire(_llm_error_event(recoverable=False), self.session, self.state)

        self.session.say.assert_called_once()
        spoken_text = self.session.say.call_args.args[0]
        self.assertIn(LLM_FAILURE_FALLBACK_LINE, spoken_text)
        self.assertIn("HOLD_MSG", spoken_text)

    def test_sets_escalation_triggered(self):
        _fire(_llm_error_event(recoverable=False), self.session, self.state)

        self.assertTrue(self.state.escalation_triggered)
        self.assertEqual(
            self.state.escalation_reason, "LLM permanently unavailable (technical failure after retries)"
        )

    def test_recoverable_error_is_ignored(self):
        _fire(_llm_error_event(recoverable=True), self.session, self.state)

        self.session.say.assert_not_called()
        self.assertFalse(self.state.escalation_triggered)

    def test_non_llm_error_is_ignored(self):
        _fire(_non_llm_error_event(), self.session, self.state)

        self.session.say.assert_not_called()
        self.assertFalse(self.state.escalation_triggered)


class TestSubsequentPermanentFailures(unittest.TestCase):
    def setUp(self):
        self.session = _make_session()
        self.state = ConversationState()
        # Simulate the first permanent failure already having happened.
        _fire(_llm_error_event(recoverable=False), self.session, self.state)
        self.session.reset_mock()

    def test_second_failure_does_not_stay_silent(self):
        _fire(_llm_error_event(recoverable=False), self.session, self.state)

        self.session.say.assert_called_once()

    def test_second_failure_speaks_short_repeat_fallback(self):
        _fire(_llm_error_event(recoverable=False), self.session, self.state)

        spoken_text = self.session.say.call_args.args[0]
        self.assertEqual(spoken_text, LLM_FAILURE_REPEAT_FALLBACK_LINE)

    def test_second_failure_does_not_repeat_full_escalation_message(self):
        _fire(_llm_error_event(recoverable=False), self.session, self.state)

        spoken_text = self.session.say.call_args.args[0]
        self.assertNotIn(LLM_FAILURE_FALLBACK_LINE, spoken_text)

    def test_second_failure_does_not_create_duplicate_escalation_event(self):
        reason_before = self.state.escalation_reason
        tool_calls_before = len(self.state.tool_calls_made)

        _fire(_llm_error_event(recoverable=False), self.session, self.state)

        self.assertTrue(self.state.escalation_triggered)  # still True, not reset
        self.assertEqual(self.state.escalation_reason, reason_before)  # unchanged, not re-set
        self.assertEqual(len(self.state.tool_calls_made), tool_calls_before)  # no new record_tool() call

    def test_third_failure_also_speaks_short_fallback_not_silence(self):
        _fire(_llm_error_event(recoverable=False), self.session, self.state)
        self.session.reset_mock()

        _fire(_llm_error_event(recoverable=False), self.session, self.state)

        self.session.say.assert_called_once_with(LLM_FAILURE_REPEAT_FALLBACK_LINE, allow_interruptions=True)

    def test_recoverable_error_after_escalation_is_still_ignored(self):
        _fire(_llm_error_event(recoverable=True), self.session, self.state)

        self.session.say.assert_not_called()


if __name__ == "__main__":
    unittest.main()

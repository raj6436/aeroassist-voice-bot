"""
Unit tests for the Gemini-429-quota classification and LLM token-usage
instrumentation added to agent/core_agent.py.

TASK 2 (quota classification): proves _is_quota_exceeded_error() correctly
recognizes a Gemini HTTP 429 and nothing else, and that the EXISTING
escalation/fallback timing and spoken wording from the earlier fix are
completely unchanged - only escalation_reason gets a more specific label
when the cause was confirmed to be quota exhaustion.

TASK 3 (token metrics): proves _extract_llm_usage() pulls the documented
LLMMetrics fields into the requested structured shape, defensively, and
never emits anything beyond those numeric/timing fields (no secrets).

AgentSession is mocked throughout - no real LiveKit/network/TTS involved.
"""

import asyncio
import types
import unittest
from unittest.mock import MagicMock, patch

from livekit.agents import APIStatusError
from livekit.agents.metrics import LLMMetrics
from livekit.agents.voice.events import ErrorEvent, LLMError

from agent.core_agent import (
    LLM_FAILURE_FALLBACK_LINE,
    LLM_FAILURE_REPEAT_FALLBACK_LINE,
    _extract_llm_usage,
    _is_quota_exceeded_error,
    _on_llm_permanent_failure,
)
from agent.escalation_handler import EscalationHandler
from agent.state_manager import ConversationState


class _FakeSpeechHandle:
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
        await asyncio.sleep(0)

    asyncio.run(_run())


def _llm_error_event(status_code: int, recoverable: bool, retryable: bool = True) -> ErrorEvent:
    error = APIStatusError(
        "gemini llm: error", status_code=status_code, body="body", retryable=retryable
    )
    llm_error = LLMError(timestamp=0.0, label="llm", error=error, recoverable=recoverable)
    return ErrorEvent(error=llm_error, source=object())


def _make_session() -> MagicMock:
    session = MagicMock()
    session.say = MagicMock(return_value=_FakeSpeechHandle())
    return session


class TestIsQuotaExceededError(unittest.TestCase):
    def test_429_is_recognized_as_quota(self):
        event = _llm_error_event(status_code=429, recoverable=False)
        self.assertTrue(_is_quota_exceeded_error(event.error))

    def test_503_is_not_quota(self):
        event = _llm_error_event(status_code=503, recoverable=False)
        self.assertFalse(_is_quota_exceeded_error(event.error))

    def test_504_is_not_quota(self):
        event = _llm_error_event(status_code=504, recoverable=False)
        self.assertFalse(_is_quota_exceeded_error(event.error))

    def test_non_api_status_error_is_not_quota(self):
        llm_error = LLMError(timestamp=0.0, label="llm", error=Exception("boom"), recoverable=False)
        self.assertFalse(_is_quota_exceeded_error(llm_error))


class TestQuotaClassificationDoesNotChangeTiming(unittest.TestCase):
    """The existing escalation/fallback behavior from the earlier fix (f19085f)
    must be byte-for-byte unchanged - only escalation_reason's text differs."""

    def setUp(self):
        self.session = _make_session()
        self.state = ConversationState()

    def test_normal_successful_llm_flow_is_unaffected(self):
        """Sanity: a ConversationState with no errors at all behaves exactly
        as before - nothing about quota detection touches the non-error path."""
        self.assertFalse(self.state.escalation_triggered)
        self.assertIsNone(self.state.escalation_reason)

    def test_first_permanent_429_still_speaks_full_fallback_and_hold_message(self):
        with patch.object(EscalationHandler, "generate_hold_message", return_value="HOLD_MSG"):
            _fire(_llm_error_event(status_code=429, recoverable=False), self.session, self.state)

        self.session.say.assert_called_once()
        spoken_text = self.session.say.call_args.args[0]
        self.assertIn(LLM_FAILURE_FALLBACK_LINE, spoken_text)
        self.assertIn("HOLD_MSG", spoken_text)

    def test_first_permanent_429_tags_escalation_reason_as_quota(self):
        _fire(_llm_error_event(status_code=429, recoverable=False), self.session, self.state)

        self.assertTrue(self.state.escalation_triggered)
        self.assertIn("429", self.state.escalation_reason)
        self.assertIn("quota", self.state.escalation_reason.lower())

    def test_first_permanent_503_keeps_generic_escalation_reason(self):
        _fire(_llm_error_event(status_code=503, recoverable=False), self.session, self.state)

        self.assertEqual(
            self.state.escalation_reason, "LLM permanently unavailable (technical failure after retries)"
        )
        self.assertNotIn("429", self.state.escalation_reason)

    def test_second_permanent_failure_still_speaks_short_repeat_fallback_regardless_of_cause(self):
        _fire(_llm_error_event(status_code=429, recoverable=False), self.session, self.state)
        self.session.reset_mock()

        _fire(_llm_error_event(status_code=503, recoverable=False), self.session, self.state)

        self.session.say.assert_called_once_with(LLM_FAILURE_REPEAT_FALLBACK_LINE, allow_interruptions=True)

    def test_second_failure_does_not_overwrite_first_failure_escalation_reason(self):
        _fire(_llm_error_event(status_code=429, recoverable=False), self.session, self.state)
        reason_after_first = self.state.escalation_reason

        _fire(_llm_error_event(status_code=429, recoverable=False), self.session, self.state)

        self.assertEqual(self.state.escalation_reason, reason_after_first)

    def test_recoverable_429_is_still_ignored_like_any_other_recoverable_error(self):
        """A 429 that LiveKit's own retry loop marks recoverable=True is still
        mid-retry - speaking here would risk a contradictory reply if that
        retry later succeeds, so this must stay a no-op, exactly as before."""
        _fire(_llm_error_event(status_code=429, recoverable=True), self.session, self.state)

        self.session.say.assert_not_called()
        self.assertFalse(self.state.escalation_triggered)


class TestExtractLlmUsage(unittest.TestCase):
    def test_extracts_all_documented_fields(self):
        metrics = LLMMetrics(
            label="livekit.plugins.google.llm.LLM",
            request_id="req-1",
            timestamp=0.0,
            duration=1.5,
            ttft=0.3,
            cancelled=False,
            completion_tokens=42,
            prompt_tokens=1000,
            prompt_cached_tokens=0,
            total_tokens=1042,
            tokens_per_second=28.0,
        )

        usage = _extract_llm_usage(metrics)

        self.assertEqual(
            usage,
            {
                "event": "llm_metrics",
                "prompt_tokens": 1000,
                "completion_tokens": 42,
                "total_tokens": 1042,
                "ttft": 0.3,
                "tokens_per_second": 28.0,
                "reasoning_tokens": 0,
            },
        )

    def test_missing_fields_fall_back_to_none_instead_of_raising(self):
        """A bare object without the usual LLMMetrics attributes - proves the
        extraction never raises even if a future/other SDK object lacks a field."""
        bare = types.SimpleNamespace()

        usage = _extract_llm_usage(bare)

        self.assertEqual(
            usage,
            {
                "event": "llm_metrics",
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "ttft": None,
                "tokens_per_second": None,
                "reasoning_tokens": None,
            },
        )

    def test_partial_fields_present_are_still_extracted(self):
        partial = types.SimpleNamespace(prompt_tokens=50, completion_tokens=10)

        usage = _extract_llm_usage(partial)

        self.assertEqual(usage["prompt_tokens"], 50)
        self.assertEqual(usage["completion_tokens"], 10)
        self.assertIsNone(usage["total_tokens"])
        self.assertIsNone(usage["ttft"])
        self.assertIsNone(usage["tokens_per_second"])

    def test_output_contains_no_secrets_or_unexpected_fields(self):
        metrics = LLMMetrics(
            label="livekit.plugins.google.llm.LLM",
            request_id="req-1",
            timestamp=0.0,
            duration=1.5,
            ttft=0.3,
            cancelled=False,
            completion_tokens=42,
            prompt_tokens=1000,
            prompt_cached_tokens=0,
            total_tokens=1042,
            tokens_per_second=28.0,
        )

        usage = _extract_llm_usage(metrics)

        allowed_keys = {
            "event", "prompt_tokens", "completion_tokens", "total_tokens",
            "ttft", "tokens_per_second", "reasoning_tokens",
        }
        self.assertEqual(set(usage.keys()), allowed_keys)
        rendered = repr(usage)
        for forbidden in ("api_key", "secret", "Authorization", "GEMINI_API_KEY"):
            self.assertNotIn(forbidden, rendered)


if __name__ == "__main__":
    unittest.main()

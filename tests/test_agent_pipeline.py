"""
Unit tests for AeroAssist agent pipeline:
- ConversationState
- EscalationHandler
- FlightTools (LiveKit Agents 1.x function tools)
"""

import unittest
from agent.state_manager import ConversationState
from agent.escalation_handler import EscalationHandler
from agent import tools as flight_tools


class _FakeRunContext:
    """Minimal stand-in for livekit.agents.RunContext, exposing only what the
    tool functions actually read (`.userdata`)."""

    def __init__(self, userdata):
        self.userdata = userdata


class TestConversationState(unittest.TestCase):
    """Test state manager and frustration scoring."""

    def test_initial_state(self):
        state = ConversationState()
        self.assertEqual(state.turn_count, 0)
        self.assertEqual(state.frustration_score, 0.0)
        self.assertFalse(state.should_escalate())

    def test_frustration_keyword_detection(self):
        state = ConversationState()
        state.update_turn("Mujhe insaan ya manager se baat karni hai!")
        self.assertGreaterEqual(state.frustration_score, 0.35)
        self.assertIn("Keyword trigger", state.escalation_reason)

    def test_escalation_threshold(self):
        state = ConversationState()
        state.update_turn("agent se baat karao")
        state.update_turn("manager ko bulao")
        self.assertGreaterEqual(state.frustration_score, 0.7)
        self.assertTrue(state.should_escalate())

    def test_repetition_escalation(self):
        state = ConversationState()
        state.update_turn("kaha hai")
        state.update_turn("kaha hai")
        state.update_turn("kaha hai")
        state.update_turn("kaha hai")
        self.assertGreaterEqual(state.unanswered_repeats, 3)
        self.assertTrue(state.should_escalate())

    def test_record_tool_and_handoff_summary(self):
        state = ConversationState(verified_pnr="6E2849", passenger_name="Rahul Sharma")
        state.record_tool("lookup_booking", "Found 6E2849")
        summary = state.get_summary_for_handoff()
        self.assertIn("Rahul Sharma", summary)
        self.assertIn("6E2849", summary)
        self.assertIn("lookup_booking", summary)


class TestEscalationHandler(unittest.TestCase):
    """Test escalation evaluation, hold announcements, and briefings."""

    def test_no_escalation_on_calm_user(self):
        state = ConversationState()
        state.update_turn("Please tell me flight status")
        self.assertIsNone(EscalationHandler.evaluate_and_escalate(state))

    def test_escalate_when_threshold_reached(self):
        state = ConversationState()
        state.escalation_triggered = True
        state.escalation_reason = "Customer requested human"
        result = EscalationHandler.evaluate_and_escalate(state)
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("ESCALATE:"))

    def test_generate_hold_message(self):
        msg = EscalationHandler.generate_hold_message()
        self.assertIn("senior flight executive", msg)
        self.assertIn("transfer", msg)

    def test_generate_agent_briefing(self):
        state = ConversationState(verified_pnr="AI805X", passenger_name="Priya Nair")
        state.record_tool("check_flight_status", "AI-805 DELAYED")
        state.escalation_reason = "Flight delayed significantly"
        briefing = EscalationHandler.generate_agent_briefing(state)
        self.assertIn("Priya Nair", briefing)
        self.assertIn("AI805X", briefing)
        self.assertIn("check_flight_status", briefing)


class TestFlightTools(unittest.IsolatedAsyncioTestCase):
    """Test the LiveKit Agents 1.x function tools in agent/tools.py."""

    def setUp(self):
        self.state = ConversationState()
        self.ctx = _FakeRunContext(self.state)

    async def test_lookup_booking_success(self):
        res = await flight_tools.lookup_booking(self.ctx, "6E2849")
        self.assertIn("Rahul Sharma", res)
        self.assertIn("6E-2049", res)
        self.assertEqual(self.state.verified_pnr, "6E2849")
        self.assertEqual(len(self.state.tool_calls_made), 1)

    async def test_lookup_booking_not_found(self):
        res = await flight_tools.lookup_booking(self.ctx, "NONEXIST")
        self.assertIn("Maaf kijiye", res)

    async def test_check_flight_status(self):
        res = await flight_tools.check_flight_status(self.ctx, "6E-2049")
        self.assertIn("6E-2049", res)
        self.assertIn("on time", res)

    async def test_calculate_reschedule_quote(self):
        res = await flight_tools.calculate_reschedule_quote(self.ctx, "6E2849", "6E-2051", "2026-09-22")
        self.assertIn("rupees", res)
        self.assertIn("reschedule fee", res)

    async def test_calculate_refund_airline_cancelled(self):
        res = await flight_tools.calculate_refund(self.ctx, "QP1102")
        self.assertIn("100 percent refund", res)

    async def test_escalate_to_human(self):
        res = await flight_tools.escalate_to_human(self.ctx, "Customer is upset about delay")
        self.assertIn("transfer", res)
        self.assertTrue(self.state.escalation_triggered)
        self.assertEqual(self.state.escalation_reason, "Customer is upset about delay")


if __name__ == "__main__":
    unittest.main()


"""
Conversation state manager for AeroAssist voice bot.
Tracks conversation context, intent, customer details, frustration levels, and escalation criteria.
"""

from typing import Optional, List, Dict, Any
from database.mock_db import get_booking_by_pnr
from services.summary_service import generate_warm_transfer_summary


# Negative / escalation keywords in Hindi, English, and Hinglish
ESCALATION_KEYWORDS = [
    "agent",
    "insaan",
    "human",
    "manager",
    "call transfer",
    "executive",
    "frustrated",
    "fed up",
    "bakwaas",
    "bakwas",
    "enough",
    "kisi aur se",
    "supervisor",
    "agent se baat karo",
    "gussa",
    "complaint",
    "cheat",
]


class ConversationState:
    """
    Manages state for an ongoing customer call session.
    """

    def __init__(
        self,
        customer_phone: Optional[str] = None,
        verified_pnr: Optional[str] = None,
        passenger_name: Optional[str] = None,
        current_intent: Optional[str] = None,
    ):
        self.customer_phone: Optional[str] = customer_phone
        self.verified_pnr: Optional[str] = verified_pnr
        self.passenger_name: Optional[str] = passenger_name
        self.current_intent: Optional[str] = current_intent
        self.turn_count: int = 0
        self.unanswered_repeats: int = 0
        self.frustration_score: float = 0.0
        self.tool_calls_made: List[Dict[str, Any]] = []
        self.escalation_triggered: bool = False
        self.escalation_reason: Optional[str] = None
        self._last_user_message: Optional[str] = None

    def update_turn(self, user_message: str) -> None:
        """
        Increment turn count, analyze sentiment / frustration keywords,
        and detect repeated queries.
        """
        self.turn_count += 1
        clean_msg = user_message.strip().lower()

        # Check for frustration / human escalation keywords
        matched_keyword = False
        for kw in ESCALATION_KEYWORDS:
            if kw in clean_msg:
                self.frustration_score = round(self.frustration_score + 0.35, 2)
                matched_keyword = True
                if not self.escalation_reason:
                    self.escalation_reason = f"Keyword trigger: '{kw}'"
                break

        # Check for repetition of short identical messages
        if self._last_user_message and clean_msg == self._last_user_message:
            self.unanswered_repeats += 1
        elif len(clean_msg) < 15 and self._last_user_message and clean_msg in self._last_user_message:
            self.unanswered_repeats += 1
        else:
            # Reset if different message
            if self.unanswered_repeats > 0 and not matched_keyword:
                self.unanswered_repeats = max(0, self.unanswered_repeats - 1)

        self._last_user_message = clean_msg

    def should_escalate(self) -> bool:
        """
        Check if conversation meets escalation criteria:
        1. Frustration score >= 0.7
        2. Explicit escalation triggered
        3. Unanswered repeats >= 3
        """
        if self.escalation_triggered:
            return True
        if self.frustration_score >= 0.7:
            if not self.escalation_reason:
                self.escalation_reason = f"High frustration score ({self.frustration_score})"
            return True
        if self.unanswered_repeats >= 3:
            if not self.escalation_reason:
                self.escalation_reason = f"Repeated unanswered requests ({self.unanswered_repeats} times)"
            return True
        return False

    def record_tool(self, tool_name: str, details: str) -> None:
        """Record an executed tool and its output details."""
        self.tool_calls_made.append({"tool": tool_name, "details": details})

    def record_tool_call(self, tool_name: str, result: str) -> None:
        """Alias for record_tool to maintain multi-standard compatibility."""
        self.record_tool(tool_name, result)

    def get_summary_for_handoff(self) -> str:
        """
        Build a comprehensive warm-transfer briefing for a human supervisor.
        """
        booking = get_booking_by_pnr(self.verified_pnr) if self.verified_pnr else None
        checks = [f"{t['tool']}: {t['details']}" for t in self.tool_calls_made]
        reason = self.escalation_reason or "Customer requested senior support"

        if booking:
            return generate_warm_transfer_summary(
                booking=booking,
                issue_summary=f"Intent: {self.current_intent or 'General inquiry'}, Turns: {self.turn_count}",
                checks_performed=checks,
                escalation_reason=reason,
            )

        # Fallback if no valid booking object is loaded yet
        tools_text = "\n".join(f"  - {c}" for c in checks) if checks else "  - None"
        return (
            "========== WARM TRANSFER BRIEFING ==========\n"
            f"Customer Name : {self.passenger_name or 'Unknown'}\n"
            f"Phone         : {self.customer_phone or 'Unknown'}\n"
            f"PNR           : {self.verified_pnr or 'Not Provided'}\n"
            f"Turns / Status: {self.turn_count} turns, Frustration: {self.frustration_score}\n"
            "---------------------------------------------\n"
            f"Issue Summary:\n  Intent: {self.current_intent or 'Flight Support'}\n"
            "---------------------------------------------\n"
            f"AI Checks Performed:\n{tools_text}\n"
            "---------------------------------------------\n"
            f"Escalation Reason:\n  {reason}\n"
            "============================================="
        )


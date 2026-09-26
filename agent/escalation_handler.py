"""
Escalation handler for AeroAssist voice bot.
Evaluates escalation criteria, generates hold announcements, and prepares briefings for human agents.
"""

from typing import Optional
from agent.state_manager import ConversationState


class EscalationHandler:
    """
    Evaluates customer frustration or explicit transfer requests
    and coordinates warm transfer workflows.
    """

    @staticmethod
    def evaluate_and_escalate(state: ConversationState) -> Optional[str]:
        """
        Check if the current state warrants escalation.
        Returns an escalation descriptor string if true, else None.
        """
        if state.should_escalate():
            reason = state.escalation_reason or "Customer requested human or high frustration detected"
            return f"ESCALATE: {reason}"
        return None

    @staticmethod
    def generate_hold_message() -> str:
        """
        Voice-friendly prompt played to the customer while transfer connects.
        """
        return "Aapki call hamare senior flight executive ko transfer ho rahi hai. Kripya line par bane rahein."

    @staticmethod
    def generate_agent_briefing(state: ConversationState) -> str:
        """
        Produce a structured briefing card for the incoming human agent.
        """
        passenger = state.passenger_name or "Unknown"
        pnr = state.verified_pnr or "Not Provided"
        reason = state.escalation_reason or "Customer requested human or high frustration detected"

        if state.tool_calls_made:
            tools_str = ", ".join([f"{t['tool']} ({t['details']})" for t in state.tool_calls_made])
        else:
            tools_str = "None"

        briefing = (
            "--- WARM TRANSFER BRIEFING ---\n"
            f"Passenger: {passenger} | PNR: {pnr}\n"
            f"Trigger: {reason}\n"
            f"Tools Called: {tools_str}\n"
            "-------------------------------"
        )
        return briefing


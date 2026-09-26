"""
LLM Function Tools for AeroAssist.
Exposes airline services as conversational, voice-friendly tools for the AI agent.
"""

from typing import Annotated, Optional
import livekit.agents.llm as llm
from services.flight_service import FlightService
from agent.state_manager import ConversationState

# Ensure compatibility across livekit-agents versions
if not hasattr(llm, "FunctionContext"):
    class _FunctionContext:
        pass
    llm.FunctionContext = _FunctionContext

if not hasattr(llm, "ai_callable"):
    def _ai_callable(f=None, **kwargs):
        if f is None:
            return lambda fn: fn
        return f
    llm.ai_callable = _ai_callable

if not hasattr(llm, "TypeInfo"):
    class _TypeInfo:
        def __init__(self, description: str = ""):
            self.description = description
    llm.TypeInfo = _TypeInfo


class FlightTools(llm.FunctionContext):
    """
    Function tools exposed to the LLM for handling airline customer queries.
    All methods return voice-friendly Hinglish responses.
    """

    def __init__(self, state: Optional[ConversationState] = None):
        super().__init__()
        self.flight_service = FlightService()
        self.state = state

    def _record(self, tool_name: str, details: str) -> None:
        if self.state:
            self.state.record_tool(tool_name, details)

    @llm.ai_callable()
    def lookup_booking(
        self,
        pnr: Annotated[str, llm.TypeInfo(description="The 6-character airline PNR")],
    ) -> str:
        """Look up flight booking details using customer PNR."""
        booking = self.flight_service.get_booking(pnr)
        if not booking:
            msg = f"Maaf kijiye, PNR {pnr} humare system mein nahi mila. Kripya ek baar dobara check karein."
            self._record("lookup_booking", f"PNR {pnr} not found")
            return msg

        # Update state if attached
        if self.state:
            self.state.verified_pnr = booking.pnr
            self.state.passenger_name = booking.passenger_name

        flight = booking.flight
        status_str = flight.status.value.lower().replace("_", " ")
        msg = (
            f"Aapki booking mil gayi hai. Passenger {booking.passenger_name}, "
            f"flight {flight.flight_number} from {flight.origin} to {flight.destination}, "
            f"status {status_str}."
        )
        self._record("lookup_booking", f"Found {booking.pnr} for {booking.passenger_name}")
        return msg

    @llm.ai_callable()
    def check_flight_status(
        self,
        flight_number: Annotated[str, llm.TypeInfo(description="The flight number e.g. 6E-2049 or AI-805")],
    ) -> str:
        """Check live operating status of a flight."""
        segment = self.flight_service.check_flight_status(flight_number)
        if not segment:
            msg = f"Maaf kijiye, flight {flight_number} ki jaankari nahi mili. Kripya flight number check karein."
            self._record("check_flight_status", f"Flight {flight_number} not found")
            return msg

        dep_time_str = segment.departure_time.strftime("%I:%M %p")
        status_val = segment.status.value.lower().replace("_", " ")
        msg = f"Flight {segment.flight_number} {status_val} hai, scheduled departure {dep_time_str}."
        self._record("check_flight_status", f"{segment.flight_number}: {status_val}")
        return msg

    @llm.ai_callable()
    def calculate_reschedule_quote(
        self,
        pnr: Annotated[str, llm.TypeInfo(description="Booking PNR")],
        new_flight_number: Annotated[str, llm.TypeInfo(description="Target flight number")],
        new_date: Annotated[str, llm.TypeInfo(description="Target date")],
    ) -> str:
        """Calculate the cost of rescheduling an existing flight booking."""
        quote = self.flight_service.calculate_reschedule_quote(pnr, new_flight_number)
        if not quote:
            msg = "Maaf kijiye, reschedule quote calculate nahi ho paya. Kripya PNR aur flight number check karein."
            self._record("calculate_reschedule_quote", f"Failed for PNR {pnr}")
            return msg

        diff = int(quote.fare_difference)
        fee = int(quote.reschedule_fee)
        total = int(quote.total_payable)
        msg = (
            f"Reschedule ke liye fare difference {diff} rupees aur reschedule fee {fee} rupees, "
            f"total {total} rupees payable hoga."
        )
        self._record("calculate_reschedule_quote", f"Quote total ₹{total} for PNR {pnr}")
        return msg

    # Alias for get_reschedule_quote
    get_reschedule_quote = calculate_reschedule_quote

    @llm.ai_callable()
    def calculate_refund(
        self,
        pnr: Annotated[str, llm.TypeInfo(description="Booking PNR")],
    ) -> str:
        """Estimate the refundable amount after cancelling a flight booking."""
        refund = self.flight_service.calculate_refund(pnr)
        if not refund:
            msg = "Maaf kijiye, refund calculate nahi ho saka. Kripya PNR dobara check karein."
            self._record("calculate_refund", f"Failed for PNR {pnr}")
            return msg

        penalty = int(refund.cancellation_penalty)
        refundable = int(refund.refundable_amount)
        if penalty == 0:
            msg = f"Flight airline dwara cancel hone ke kaaran aapko poora 100 percent refund yaani {refundable} rupees milega."
        else:
            msg = f"Cancellation penalty {penalty} rupees ke baad aapko {refundable} rupees refund milega."

        self._record("calculate_refund", f"Refund ₹{refundable}, penalty ₹{penalty}")
        return msg

    # Alias for get_cancellation_refund
    get_cancellation_refund = calculate_refund

    @llm.ai_callable()
    def escalate_to_human(
        self,
        reason: Annotated[str, llm.TypeInfo(description="Reason for escalating to human agent")],
    ) -> str:
        """Initiate warm escalation to a senior customer support executive."""
        if self.state:
            self.state.escalation_triggered = True
            self.state.escalation_reason = reason
        self._record("escalate_to_human", f"Triggered: {reason}")
        return "Main aapki call turant humare human flight executive ko transfer kar rahi hoon. Kripya hold karein."


# AssistantFnc alias for alternate pattern compatibility
AssistantFnc = FlightTools


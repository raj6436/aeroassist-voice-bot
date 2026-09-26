"""
LLM Function Tools for AeroAssist.
Exposes airline services as conversational, voice-friendly tools for the AI agent
using the real LiveKit Agents 1.x (`livekit-agents==1.8.2`) function-calling API.
"""

from livekit.agents import RunContext, function_tool

from agent.escalation_handler import EscalationHandler
from agent.state_manager import ConversationState
from services.flight_service import FlightService

flight_service = FlightService()


@function_tool
async def lookup_booking(context: RunContext[ConversationState], pnr: str) -> str:
    """Look up flight booking details using the customer's PNR.

    Args:
        pnr: The 6-character airline PNR / booking reference.
    """
    state = context.userdata
    booking = flight_service.get_booking(pnr)
    if not booking:
        state.record_tool("lookup_booking", f"PNR {pnr} not found")
        return f"Maaf kijiye, PNR {pnr} humare system mein nahi mila. Kripya ek baar dobara check karein."

    state.verified_pnr = booking.pnr
    state.passenger_name = booking.passenger_name

    flight = booking.flight
    status_str = flight.status.value.lower().replace("_", " ")
    msg = (
        f"Aapki booking mil gayi hai. Passenger {booking.passenger_name}, "
        f"flight {flight.flight_number} from {flight.origin} to {flight.destination}, "
        f"status {status_str}."
    )
    state.record_tool("lookup_booking", f"Found {booking.pnr} for {booking.passenger_name}")
    return msg


@function_tool
async def check_flight_status(context: RunContext[ConversationState], flight_number: str) -> str:
    """Check the live operating status of a flight.

    Args:
        flight_number: The airline flight number, e.g. 6E-2049 or AI-805.
    """
    state = context.userdata
    segment = flight_service.check_flight_status(flight_number)
    if not segment:
        state.record_tool("check_flight_status", f"Flight {flight_number} not found")
        return f"Maaf kijiye, flight {flight_number} ki jaankari nahi mili. Kripya flight number check karein."

    dep_time_str = segment.departure_time.strftime("%I:%M %p")
    status_val = segment.status.value.lower().replace("_", " ")
    state.record_tool("check_flight_status", f"{segment.flight_number}: {status_val}")
    return f"Flight {segment.flight_number} {status_val} hai, scheduled departure {dep_time_str} hai."


@function_tool
async def calculate_reschedule_quote(
    context: RunContext[ConversationState],
    pnr: str,
    new_flight_number: str,
    new_date: str = "",
) -> str:
    """Calculate the cost of rescheduling an existing flight booking.

    Args:
        pnr: The customer's existing booking PNR.
        new_flight_number: Target flight number to reschedule to, e.g. 6E-2051.
        new_date: Optional desired date for the new flight.
    """
    state = context.userdata
    quote = flight_service.calculate_reschedule_quote(pnr, new_flight_number)
    if not quote:
        state.record_tool("calculate_reschedule_quote", f"Failed for PNR {pnr}")
        return "Maaf kijiye, reschedule quote calculate nahi ho paya. Kripya PNR aur flight number check karein."

    diff = int(quote.fare_difference)
    fee = int(quote.reschedule_fee)
    total = int(quote.total_payable)
    state.record_tool("calculate_reschedule_quote", f"Quote total ₹{total} for PNR {pnr}")
    return (
        f"Reschedule ke liye fare difference {diff} rupees aur reschedule fee {fee} rupees, "
        f"total {total} rupees payable hoga."
    )


@function_tool
async def calculate_refund(context: RunContext[ConversationState], pnr: str) -> str:
    """Estimate the refundable amount after cancelling a flight booking.

    Args:
        pnr: The customer's booking PNR to estimate the refund for.
    """
    state = context.userdata
    refund = flight_service.calculate_refund(pnr)
    if not refund:
        state.record_tool("calculate_refund", f"Failed for PNR {pnr}")
        return "Maaf kijiye, refund calculate nahi ho saka. Kripya PNR dobara check karein."

    penalty = int(refund.cancellation_penalty)
    refundable = int(refund.refundable_amount)
    if penalty == 0:
        msg = f"Flight airline dwara cancel hone ke kaaran aapko poora 100 percent refund yaani {refundable} rupees milega."
    else:
        msg = f"Cancellation penalty {penalty} rupees ke baad aapko {refundable} rupees refund milega."

    state.record_tool("calculate_refund", f"Refund ₹{refundable}, penalty ₹{penalty}")
    return msg


@function_tool
async def escalate_to_human(context: RunContext[ConversationState], reason: str) -> str:
    """Initiate a warm escalation to a senior customer support executive.

    Args:
        reason: The reason for escalating this call to a human supervisor.
    """
    state = context.userdata
    state.escalation_triggered = True
    state.escalation_reason = reason
    state.record_tool("escalate_to_human", f"Triggered: {reason}")
    return EscalationHandler.generate_hold_message()


FLIGHT_TOOLS = [
    lookup_booking,
    check_flight_status,
    calculate_reschedule_quote,
    calculate_refund,
    escalate_to_human,
]

"""
Warm-transfer summary generator.

Produces a structured briefing text that a human agent can quickly scan
when the AI escalates a call.
"""

from database.schema import BookingRecord


def generate_warm_transfer_summary(
    booking: BookingRecord,
    issue_summary: str,
    checks_performed: list[str],
    escalation_reason: str,
) -> str:
    """
    Build a concise warm-transfer briefing for the human agent.

    Parameters
    ----------
    booking : BookingRecord
        The customer's booking record.
    issue_summary : str
        One-line summary of the customer's issue.
    checks_performed : list[str]
        What the AI already looked up / verified.
    escalation_reason : str
        Why the call is being escalated.

    Returns
    -------
    str
        Formatted multi-line briefing text.
    """
    checks_text = "\n".join(f"  - {check}" for check in checks_performed) if checks_performed else "  - None"

    summary = (
        "========== WARM TRANSFER BRIEFING ==========\n"
        f"Customer Name : {booking.passenger_name}\n"
        f"Phone         : {booking.phone}\n"
        f"PNR           : {booking.pnr}\n"
        f"Flight        : {booking.flight.flight_number} "
        f"({booking.flight.origin} → {booking.flight.destination})\n"
        f"Flight Status : {booking.flight.status.value}\n"
        f"Booking Status: {booking.booking_status.value}\n"
        "---------------------------------------------\n"
        f"Issue Summary:\n  {issue_summary}\n"
        "---------------------------------------------\n"
        f"AI Checks Performed:\n{checks_text}\n"
        "---------------------------------------------\n"
        f"Escalation Reason:\n  {escalation_reason}\n"
        "============================================="
    )
    return summary


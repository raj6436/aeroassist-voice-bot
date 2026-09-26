"""
Flight booking service layer.

Encapsulates all business logic for PNR lookups, flight status,
reschedule quotes, and refund calculations.
"""

from datetime import datetime
from typing import Optional

from database.mock_db import get_booking_by_pnr, get_flight_by_number, FLIGHTS
from database.schema import (
    BookingRecord,
    FlightSegment,
    FlightStatus,
    RescheduleQuote,
    RefundEstimate,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STANDARD_CANCELLATION_FEE = 2500.0   # INR
RESCHEDULE_FEE = 500.0               # INR


class FlightService:
    """Core business-logic service for the voice-bot backend."""

    # ── PNR lookup ────────────────────────────────────────────────────────
    @staticmethod
    def get_booking(pnr: str) -> Optional[BookingRecord]:
        """
        Retrieve a booking by PNR (case-insensitive).

        Returns None when the PNR is not found.
        """
        return get_booking_by_pnr(pnr)

    # ── Flight status ─────────────────────────────────────────────────────
    @staticmethod
    def check_flight_status(flight_number: str) -> Optional[FlightSegment]:
        """
        Look up the current status of a flight by its flight number.

        Returns None when the flight number is unknown.
        """
        return get_flight_by_number(flight_number)

    # ── Reschedule quote ──────────────────────────────────────────────────
    @staticmethod
    def calculate_reschedule_quote(
        pnr: str,
        new_flight_number: str,
        new_date: Optional[datetime] = None,
    ) -> Optional[RescheduleQuote]:
        """
        Calculate the cost of rescheduling *pnr* to *new_flight_number*.

        Parameters
        ----------
        pnr : str
            Existing booking PNR.
        new_flight_number : str
            Flight number of the desired new flight.
        new_date : datetime, optional
            Ignored in the mock – included for API completeness.

        Returns
        -------
        RescheduleQuote | None
            Cost breakdown, or None if PNR / flight is invalid.
        """
        booking = get_booking_by_pnr(pnr)
        if booking is None:
            return None

        new_flight = get_flight_by_number(new_flight_number)
        if new_flight is None:
            return None

        old_fare = booking.fare_paid
        # Simulate a new fare based on the new flight (mock: use a fixed base
        # fare proportional to the existing one for simplicity).
        new_fare = old_fare  # same-route baseline
        # If the new flight is on a different route or airline, tweak slightly
        if new_flight.flight_number != booking.flight.flight_number:
            new_fare = round(old_fare * 1.10, 2)  # 10 % higher for demo

        fare_difference = round(new_fare - old_fare, 2)
        total_payable = max(fare_difference + RESCHEDULE_FEE, 0)

        return RescheduleQuote(
            pnr=booking.pnr,
            old_flight=booking.flight,
            new_flight=new_flight,
            fare_difference=fare_difference,
            reschedule_fee=RESCHEDULE_FEE,
            total_payable=round(total_payable, 2),
        )

    # ── Refund / cancellation ─────────────────────────────────────────────
    @staticmethod
    def calculate_refund(pnr: str, reason: str = "") -> Optional[RefundEstimate]:
        """
        Estimate the refund for cancelling booking *pnr*.

        Rules
        -----
        * If the airline itself cancelled the flight → 100 % refund.
        * Otherwise → standard cancellation fee of ₹2 500 is deducted.

        Parameters
        ----------
        pnr : str
            Booking PNR.
        reason : str
            Free-text reason (used for logging; the key factor is whether
            the *flight* is marked CANCELLED by the airline).

        Returns
        -------
        RefundEstimate | None
            Refund breakdown, or None if PNR is invalid.
        """
        booking = get_booking_by_pnr(pnr)
        if booking is None:
            return None

        total_fare = booking.fare_paid

        # Airline-initiated cancellation → full refund
        if booking.flight.status == FlightStatus.CANCELLED:
            return RefundEstimate(
                pnr=booking.pnr,
                total_fare=total_fare,
                cancellation_penalty=0.0,
                refundable_amount=total_fare,
                refund_mode="Original Payment Method",
            )

        # Passenger-initiated cancellation → standard penalty
        penalty = min(STANDARD_CANCELLATION_FEE, total_fare)
        refundable = round(total_fare - penalty, 2)

        return RefundEstimate(
            pnr=booking.pnr,
            total_fare=total_fare,
            cancellation_penalty=penalty,
            refundable_amount=refundable,
            refund_mode="Original Payment Method",
        )


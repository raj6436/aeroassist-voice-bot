"""
Unit tests for FlightService business logic.
"""

import unittest
from services.flight_service import FlightService


class TestGetBooking(unittest.TestCase):
    """Tests for PNR lookup."""

    def test_valid_pnr_returns_booking(self):
        """A known PNR should return the corresponding BookingRecord."""
        booking = FlightService.get_booking("6E2849")
        self.assertIsNotNone(booking)
        self.assertEqual(booking.pnr, "6E2849")
        self.assertEqual(booking.passenger_name, "Rahul Sharma")

    def test_valid_pnr_case_insensitive(self):
        """PNR lookup must be case-insensitive."""
        booking = FlightService.get_booking("6e2849")
        self.assertIsNotNone(booking)
        self.assertEqual(booking.pnr, "6E2849")

    def test_invalid_pnr_returns_none(self):
        """An unknown PNR should return None."""
        booking = FlightService.get_booking("XXXXXX")
        self.assertIsNone(booking)


class TestCheckFlightStatus(unittest.TestCase):
    """Tests for flight status retrieval."""

    def test_known_flight_returns_segment(self):
        """A known flight number should return a FlightSegment."""
        segment = FlightService.check_flight_status("6E-2049")
        self.assertIsNotNone(segment)
        self.assertEqual(segment.flight_number, "6E-2049")
        self.assertEqual(segment.airline, "IndiGo")
        self.assertEqual(segment.origin, "DEL")
        self.assertEqual(segment.destination, "BOM")

    def test_delayed_flight_status(self):
        """AI-805 should be marked DELAYED."""
        segment = FlightService.check_flight_status("AI-805")
        self.assertIsNotNone(segment)
        self.assertEqual(segment.status.value, "DELAYED")

    def test_unknown_flight_returns_none(self):
        """An unknown flight number should return None."""
        segment = FlightService.check_flight_status("XX-0000")
        self.assertIsNone(segment)


class TestRefundCalculation(unittest.TestCase):
    """Tests for cancellation / refund logic."""

    def test_normal_cancellation_deducts_penalty(self):
        """Passenger-initiated cancel → ₹2500 penalty deducted."""
        refund = FlightService.calculate_refund("6E2849", reason="personal")
        self.assertIsNotNone(refund)
        self.assertEqual(refund.cancellation_penalty, 2500.0)
        self.assertEqual(refund.refundable_amount, 5450.0 - 2500.0)

    def test_airline_cancelled_full_refund(self):
        """Flight cancelled by airline → 100 % refund, zero penalty."""
        refund = FlightService.calculate_refund("QP1102", reason="airline cancelled")
        self.assertIsNotNone(refund)
        self.assertEqual(refund.cancellation_penalty, 0.0)
        self.assertEqual(refund.refundable_amount, 6200.0)

    def test_refund_invalid_pnr(self):
        """Refund for an unknown PNR should return None."""
        refund = FlightService.calculate_refund("BADPNR")
        self.assertIsNone(refund)


class TestRescheduleQuote(unittest.TestCase):
    """Tests for reschedule quote generation."""

    def test_valid_reschedule_quote(self):
        """Valid PNR + valid new flight → quote with fare diff + ₹500 fee."""
        quote = FlightService.calculate_reschedule_quote("6E2849", "6E-2051")
        self.assertIsNotNone(quote)
        self.assertEqual(quote.pnr, "6E2849")
        self.assertEqual(quote.reschedule_fee, 500.0)
        # fare diff should be 10 % of 5450 = 545, total = 545 + 500 = 1045
        self.assertAlmostEqual(quote.fare_difference, 545.0, places=2)
        self.assertAlmostEqual(quote.total_payable, 1045.0, places=2)

    def test_reschedule_invalid_pnr(self):
        """Reschedule with unknown PNR should return None."""
        quote = FlightService.calculate_reschedule_quote("NOPNR", "6E-2051")
        self.assertIsNone(quote)

    def test_reschedule_invalid_flight(self):
        """Reschedule to an unknown flight should return None."""
        quote = FlightService.calculate_reschedule_quote("6E2849", "XX-9999")
        self.assertIsNone(quote)


if __name__ == "__main__":
    unittest.main()


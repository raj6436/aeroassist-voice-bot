"""
In-memory mock database with realistic Indian domestic flight bookings.

Provides lookup helpers used by the service layer.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional

from database.schema import (
    BookingRecord,
    BookingStatus,
    FlightSegment,
    FlightStatus,
)

# ---------------------------------------------------------------------------
# Helper: build datetimes relative to "today" so the data always feels current
# ---------------------------------------------------------------------------
_NOW = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


def _dt(day_offset: int, hour: int, minute: int = 0) -> datetime:
    """Return a datetime *day_offset* days from today at the given hour:minute."""
    return _NOW + timedelta(days=day_offset, hours=hour, minutes=minute)


# ---------------------------------------------------------------------------
# Flight segments (re-used inside bookings and also queryable independently)
# ---------------------------------------------------------------------------
FLIGHTS: Dict[str, FlightSegment] = {
    "6E-2049": FlightSegment(
        flight_number="6E-2049",
        airline="IndiGo",
        origin="DEL",
        destination="BOM",
        departure_time=_dt(1, 6, 30),   # tomorrow 06:30
        arrival_time=_dt(1, 8, 45),
        status=FlightStatus.ON_TIME,
    ),
    "AI-805": FlightSegment(
        flight_number="AI-805",
        airline="Air India",
        origin="BLR",
        destination="GOA",
        departure_time=_dt(2, 14, 0),   # day-after-tomorrow 14:00
        arrival_time=_dt(2, 15, 20),
        status=FlightStatus.DELAYED,     # delayed by ~2 hrs
    ),
    "QP-1102": FlightSegment(
        flight_number="QP-1102",
        airline="Akasa Air",
        origin="CCU",
        destination="DEL",
        departure_time=_dt(3, 9, 15),
        arrival_time=_dt(3, 11, 45),
        status=FlightStatus.CANCELLED,   # cancelled by airline
    ),
    "6E-8834": FlightSegment(
        flight_number="6E-8834",
        airline="IndiGo",
        origin="HYD",
        destination="MAA",
        departure_time=_dt(0, 23, 55),   # today 23:55 — departure within 24 hrs
        arrival_time=_dt(1, 1, 10),
        status=FlightStatus.ON_TIME,
    ),
    # Alternate flight used for reschedule scenarios
    "6E-2051": FlightSegment(
        flight_number="6E-2051",
        airline="IndiGo",
        origin="DEL",
        destination="BOM",
        departure_time=_dt(2, 10, 0),
        arrival_time=_dt(2, 12, 15),
        status=FlightStatus.SCHEDULED,
    ),
}

# ---------------------------------------------------------------------------
# Bookings keyed by PNR (uppercase)
# ---------------------------------------------------------------------------
BOOKINGS: Dict[str, BookingRecord] = {
    "6E2849": BookingRecord(
        pnr="6E2849",
        passenger_name="Rahul Sharma",
        phone="+91-9876543210",
        flight=FLIGHTS["6E-2049"],
        seat="14A",
        fare_paid=5450.0,
        booking_status=BookingStatus.CONFIRMED,
    ),
    "AI805X": BookingRecord(
        pnr="AI805X",
        passenger_name="Priya Nair",
        phone="+91-8765432109",
        flight=FLIGHTS["AI-805"],
        seat="22C",
        fare_paid=7800.0,
        booking_status=BookingStatus.CONFIRMED,
    ),
    "QP1102": BookingRecord(
        pnr="QP1102",
        passenger_name="Amit Verma",
        phone="+91-7654321098",
        flight=FLIGHTS["QP-1102"],
        seat="3F",
        fare_paid=6200.0,
        booking_status=BookingStatus.CONFIRMED,
    ),
    "6E9921": BookingRecord(
        pnr="6E9921",
        passenger_name="Sneha Reddy",
        phone="+91-6543210987",
        flight=FLIGHTS["6E-8834"],
        seat="9B",
        fare_paid=3900.0,
        booking_status=BookingStatus.CONFIRMED,
    ),
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_booking_by_pnr(pnr: str) -> Optional[BookingRecord]:
    """Case-insensitive PNR lookup."""
    return BOOKINGS.get(pnr.upper())


def get_flight_by_number(flight_number: str) -> Optional[FlightSegment]:
    """Case-insensitive flight number lookup."""
    return FLIGHTS.get(flight_number.upper())


def list_all_bookings() -> list[BookingRecord]:
    """Return every booking in the mock DB."""
    return list(BOOKINGS.values())


"""
Pydantic models for flight booking data.

Defines the core domain objects: flights, bookings, reschedule quotes, and refund estimates.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FlightStatus(str, Enum):
    """Current operational status of a flight."""
    SCHEDULED = "SCHEDULED"
    ON_TIME = "ON_TIME"
    DELAYED = "DELAYED"
    CANCELLED = "CANCELLED"


class BookingStatus(str, Enum):
    """Current status of a passenger booking."""
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"
    RESCHEDULED = "RESCHEDULED"


class FlightSegment(BaseModel):
    """Represents a single flight leg."""
    flight_number: str = Field(..., description="Airline flight number, e.g. 6E-2049")
    airline: str = Field(..., description="Airline name, e.g. IndiGo")
    origin: str = Field(..., description="IATA origin airport code, e.g. DEL")
    destination: str = Field(..., description="IATA destination airport code, e.g. BOM")
    departure_time: datetime = Field(..., description="Scheduled departure time")
    arrival_time: datetime = Field(..., description="Scheduled arrival time")
    status: FlightStatus = Field(default=FlightStatus.SCHEDULED, description="Flight status")


class BookingRecord(BaseModel):
    """A passenger's flight booking."""
    pnr: str = Field(..., description="PNR / booking reference, e.g. 6E2849")
    passenger_name: str = Field(..., description="Full name of the passenger")
    phone: str = Field(..., description="Contact phone number")
    flight: FlightSegment = Field(..., description="Associated flight segment")
    seat: str = Field(..., description="Assigned seat, e.g. 14A")
    fare_paid: float = Field(..., ge=0, description="Total fare paid in INR")
    booking_status: BookingStatus = Field(
        default=BookingStatus.CONFIRMED, description="Booking status"
    )


class RescheduleQuote(BaseModel):
    """Cost breakdown for rescheduling a booking to a different flight."""
    pnr: str
    old_flight: FlightSegment
    new_flight: FlightSegment
    fare_difference: float = Field(..., description="New fare minus old fare (can be negative)")
    reschedule_fee: float = Field(default=500.0, description="Airline reschedule fee in INR")
    total_payable: float = Field(..., description="Net amount the passenger must pay")


class RefundEstimate(BaseModel):
    """Refund calculation after cancellation."""
    pnr: str
    total_fare: float = Field(..., ge=0, description="Original fare paid in INR")
    cancellation_penalty: float = Field(..., ge=0, description="Penalty deducted in INR")
    refundable_amount: float = Field(..., ge=0, description="Amount refunded in INR")
    refund_mode: str = Field(
        default="Original Payment Method",
        description="How the refund will be processed",
    )


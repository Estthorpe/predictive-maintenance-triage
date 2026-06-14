"""
Data contract for a single sensor reading.

A "data contract" = strict rules that incoming data MUST satisfy before it is
allowed into the system. Pydantic enforces these rules automatically: if a
reading breaks a rule, Pydantic raises a clear error instead of letting bad
data through silently.

This file defines the contract for ONE reading. Whole-batch validation is
built on top of this in a later step.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


# The set of sensor types P7 knows about. Anything outside this list is
# rejected as an unknown/garbage type. Using an Enum means the allowed values
# live in ONE place and can never be mistyped elsewhere.
class SensorType(str, Enum):
    VIBRATION = "vibration"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    RPM = "rpm"


# Physically valid ranges per sensor type: (minimum, maximum).
# A reading outside its range is the classic signature of a FAILING sensor,
# so we reject it at the door. These are sensible engineering defaults and are
# intentionally easy to adjust in one place.
VALID_RANGES: dict[SensorType, tuple[float, float]] = {
    SensorType.VIBRATION: (0.0, 100.0),  # vibration cannot be negative
    SensorType.TEMPERATURE: (-40.0, 200.0),  # degrees Celsius
    SensorType.PRESSURE: (0.0, 1000.0),  # kPa, cannot be negative
    SensorType.RPM: (0.0, 30000.0),  # revolutions/min, cannot be negative
}


class SensorReading(BaseModel):
    """One reading, from one sensor, at one moment in time."""

    # device_id: must be present and non-empty. min_length=1 forbids "".
    device_id: str = Field(min_length=1, description="ID of the device that produced the reading")

    # timestamp: when the reading was taken. Pydantic enforces it is a real datetime.
    timestamp: datetime = Field(description="When the reading was taken")

    # sensor_type: must be one of the SensorType values above; anything else is rejected.
    sensor_type: SensorType = Field(description="The kind of sensor")

    # value: the measurement itself. Range is checked below against VALID_RANGES.
    value: float = Field(description="The measured value")

    # A custom check that needs more than one field: it compares `value`
    # against the range for that specific `sensor_type`.
    @field_validator("value")
    @classmethod
    def value_must_be_in_physical_range(cls, v: float, info):
        sensor_type = info.data.get("sensor_type")
        # If sensor_type itself failed validation, skip (that error is already raised).
        if sensor_type is None:
            return v
        low, high = VALID_RANGES[sensor_type]
        if not (low <= v <= high):
            raise ValueError(
                f"{sensor_type.value} value {v} is outside valid range [{low}, {high}] "
                f"- likely a failing sensor"
            )
        return v

"""
Tests for the SensorReading data contract.

A good test for a validator checks BOTH directions:
  - valid data is accepted
  - each kind of invalid data is rejected
For a data contract, the rejection behaviour is the whole point, so we test it
deliberately.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.ingestion.contracts import SensorReading


def test_good_reading_is_accepted():
    """A valid reading should pass and keep its values."""
    reading = SensorReading(
        device_id="PUMP_07",
        timestamp=datetime(2026, 6, 14, 10, 30, 0),
        sensor_type="vibration",
        value=12.5,
    )
    assert reading.device_id == "PUMP_07"
    assert reading.value == 12.5


def test_out_of_range_value_is_rejected():
    """A physically impossible value is the signature of a failing sensor -> reject."""
    with pytest.raises(ValidationError):
        SensorReading(
            device_id="PUMP_07",
            timestamp=datetime(2026, 6, 14, 10, 30, 0),
            sensor_type="vibration",
            value=99999.0,  # way outside [0, 100]
        )


def test_empty_device_id_is_rejected():
    """A reading with no source device is useless -> reject."""
    with pytest.raises(ValidationError):
        SensorReading(
            device_id="",  # empty, breaks min_length=1
            timestamp=datetime(2026, 6, 14, 10, 30, 0),
            sensor_type="temperature",
            value=50.0,
        )


def test_unknown_sensor_type_is_rejected():
    """A sensor type we do not recognise is garbage -> reject."""
    with pytest.raises(ValidationError):
        SensorReading(
            device_id="PUMP_07",
            timestamp=datetime(2026, 6, 14, 10, 30, 0),
            sensor_type="laser_beam",  # not in SensorType
            value=10.0,
        )


def test_temperature_below_range_is_rejected():
    """A temperature below the allowed range is invalid -> reject."""
    with pytest.raises(ValidationError):
        SensorReading(
            device_id="PUMP_07",
            timestamp=datetime(2026, 6, 14, 10, 30, 0),
            sensor_type="temperature",
            value=-100.0,  # below -40
        )

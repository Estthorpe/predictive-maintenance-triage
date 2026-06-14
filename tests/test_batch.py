"""Tests for validate_batch (micro-batch validation, threshold + report policy)."""

from datetime import datetime

from src.ingestion.batch import validate_batch


def _good():
    """A reading that passes the contract."""
    return {
        "device_id": "PUMP_07",
        "timestamp": datetime(2026, 6, 14, 10, 30, 0),
        "sensor_type": "vibration",
        "value": 12.5,
    }


def _bad():
    """A reading that fails the contract (impossible vibration value)."""
    return {
        "device_id": "PUMP_07",
        "timestamp": datetime(2026, 6, 14, 10, 30, 0),
        "sensor_type": "vibration",
        "value": 99999.0,
    }


def test_all_good_batch_is_accepted():
    result = validate_batch([_good() for _ in range(5)])
    assert result.accepted is True
    assert result.valid_count == 5
    assert result.invalid_count == 0


def test_mixed_batch_within_threshold_is_accepted():
    # 1 bad out of 20 = 5% failure, under the 10% threshold
    result = validate_batch([_good() for _ in range(19)] + [_bad()])
    assert result.accepted is True
    assert result.valid_count == 19
    assert result.invalid_count == 1


def test_batch_over_threshold_is_rejected():
    # 2 bad out of 4 = 50% failure, over the 10% threshold
    result = validate_batch([_good(), _good(), _bad(), _bad()])
    assert result.accepted is False
    assert result.invalid_count == 2


def test_empty_batch_is_rejected():
    result = validate_batch([])
    assert result.accepted is False
    assert result.total == 0


def test_quarantined_reading_keeps_a_reason():
    result = validate_batch([_good() for _ in range(19)] + [_bad()])
    assert result.invalid_count == 1
    assert result.quarantined[0].reason != ""


def test_batch_at_exactly_threshold_is_accepted():
    # 1 bad out of 10 = 10% failure, exactly at the threshold
    result = validate_batch([_good() for _ in range(9)] + [_bad()])
    assert result.accepted is True
    assert result.invalid_count == 1

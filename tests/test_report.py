"""Tests for the grounded report generator.

We do NOT call the real LLM in tests (no network, no cost, deterministic):
we test the fallback path and that output is grounded in the inputs.
"""

import os
from unittest.mock import patch

from src.genai.report import _fallback_summary, generate_report


def test_fallback_summary_is_grounded():
    s = _fallback_summary("BEARING_07", 0.67, ["fft_energy", "roll_max"], "Inspect soon")
    # Must contain ONLY/EXACTLY the provided facts.
    assert "BEARING_07" in s
    assert "0.67" in s
    assert "fft_energy" in s
    assert "Inspect soon" in s


def test_fallback_handles_empty_sensors():
    s = _fallback_summary("X", 0.1, [], "No action")
    assert "none" in s


def test_generate_report_uses_fallback_without_key():
    # With no API key, it must use the grounded fallback, not crash.
    with patch.dict(os.environ, {}, clear=True):
        result = generate_report("BEARING_07", 0.67, ["fft_energy"], "Inspect soon")
    assert result["source"] == "fallback"
    assert "BEARING_07" in result["summary"]
    assert result["prompt_version"] == "v1"


def test_report_always_returns_a_summary():
    with patch.dict(os.environ, {}, clear=True):
        result = generate_report("X", 0.5, ["roll_mean"], "Monitor")
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 0

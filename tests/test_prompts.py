"""Tests for the bounded prompt template."""

from src.genai.prompts import (
    PROMPT_VERSION,
    build_maintenance_prompt,
)


def test_prompt_includes_the_real_values():
    prompt = build_maintenance_prompt(
        device_id="BEARING_07",
        risk_score=0.67,
        top_sensors=["fft_energy", "roll_max"],
        recommended_action="Inspect soon",
    )
    # The model''s ACTUAL values must appear in the prompt (grounding).
    assert "BEARING_07" in prompt
    assert "0.67" in prompt
    assert "fft_energy" in prompt
    assert "Inspect soon" in prompt


def test_prompt_includes_grounding_rules():
    prompt = build_maintenance_prompt("X", 0.1, ["roll_mean"], "No action")
    # The anti-hallucination instructions must be present.
    assert "ONLY" in prompt
    assert "do not" in prompt.lower() or "do NOT" in prompt


def test_prompt_handles_empty_sensors():
    prompt = build_maintenance_prompt("X", 0.1, [], "No action")
    assert "none provided" in prompt


def test_prompt_version_is_defined():
    assert PROMPT_VERSION == "v1"

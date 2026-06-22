"""
Grounded maintenance-report generator for P7.

Two paths, both GROUNDED in the same model output:
  1. LLM path (primary): bounded prompt -> Claude (Haiku) -> plain-language summary.
  2. Fallback path: a deterministic template summary, used when the LLM is
     unavailable (missing key, network/API error). Graceful degradation - the
     system always returns a useful, grounded report.

GenAI AUGMENTS, never replaces: the model decides the risk; the LLM only
explains it. The risk_score is passed through unchanged.
"""

import os

from src.genai.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_maintenance_prompt,
)

LLM_MODEL = "claude-haiku-4-5-20251001"


def _fallback_summary(
    device_id: str, risk_score: float, top_sensors: list[str], recommended_action: str
) -> str:
    """Deterministic, grounded summary used when the LLM is unavailable.

    GROUNDED: uses ONLY the values passed in, invents nothing.
    """
    sensors = ", ".join(top_sensors) or "none"
    return (
        f"Device {device_id} has a risk score of {risk_score:.2f}. "
        f"Top contributing signals: {sensors}. "
        f"Recommended action: {recommended_action}."
    )


def generate_report(
    device_id: str,
    risk_score: float,
    top_sensors: list[str],
    recommended_action: str,
) -> dict:
    """Produce a grounded maintenance report.

    Tries the LLM; on ANY failure, falls back to the deterministic summary.
    Returns the summary plus metadata (source path + prompt version) so every
    report is traceable.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")

    # No key -> straight to the grounded fallback (graceful degradation).
    if not api_key:
        return {
            "summary": _fallback_summary(device_id, risk_score, top_sensors, recommended_action),
            "source": "fallback",
            "prompt_version": PROMPT_VERSION,
        }

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        prompt = build_maintenance_prompt(device_id, risk_score, top_sensors, recommended_action)
        message = client.messages.create(
            model=LLM_MODEL,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        summary = message.content[0].text.strip()
        return {
            "summary": summary,
            "source": "llm",
            "prompt_version": PROMPT_VERSION,
        }
    except Exception:
        # ANY failure (network, API, key) -> grounded fallback. Never crash.
        return {
            "summary": _fallback_summary(device_id, risk_score, top_sensors, recommended_action),
            "source": "fallback",
            "prompt_version": PROMPT_VERSION,
        }

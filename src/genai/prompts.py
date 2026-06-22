"""
Versioned, bounded prompt templates for P7 GenAI report generation.

A prompt is CODE: changing it changes the output, so prompts are versioned
artifacts (PROMPT_VERSION) for traceability and reproducibility.

The template is BOUNDED to keep the LLM GROUNDED - it may only translate the
model''s structured output into prose, never invent data. Each rule below blocks
a specific hallucination failure mode.
"""

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are a maintenance assistant. You translate a predictive-maintenance "
    "model''s structured output into a short, plain-language summary for a "
    "maintenance supervisor. You are an interpreter, not an analyst: you explain "
    "the form of the data, never add to its content."
)

# The grounding rules - each forbids a specific way an LLM can drift from facts.
GROUNDING_RULES = (
    "STRICT GROUNDING RULES:\n"
    "1. Use ONLY the values provided below. Do not use outside knowledge.\n"
    "2. Do NOT state any sensor values, frequencies, temperatures, or "
    "measurements that are not in the provided data.\n"
    "3. Base the recommended action ONLY on the risk level given; do NOT invent "
    "part numbers, procedures, or specifications.\n"
    "4. Do NOT change or reinterpret the risk score; explain it as given.\n"
    "5. If information needed for the summary is missing, say so explicitly "
    "rather than guessing.\n"
)


def build_maintenance_prompt(
    device_id: str,
    risk_score: float,
    top_sensors: list[str],
    recommended_action: str,
) -> str:
    """Assemble the bounded prompt from the model''s ACTUAL structured output.

    The model''s real values are injected; the LLM is constrained to explain
    only these. Returns the full user-prompt string.
    """
    sensors = ", ".join(top_sensors) if top_sensors else "none provided"
    return (
        f"{GROUNDING_RULES}\n"
        "MODEL OUTPUT (the only facts you may use):\n"
        f"- device_id: {device_id}\n"
        f"- risk_score: {risk_score} (0 = healthy, 1 = high risk)\n"
        f"- top_contributing_features: {sensors}\n"
        f"- recommended_action: {recommended_action}\n\n"
        "TASK: Write a 2-3 sentence maintenance summary for a supervisor that "
        "explains what this risk score means and references the contributing "
        "features by name. Follow the grounding rules exactly."
    )

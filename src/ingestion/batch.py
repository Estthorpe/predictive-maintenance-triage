"""
Batch validation for micro-batches of sensor readings.

Real IoT data arrives in irregular bursts (micro-batches), not one reading at a
time, and a burst may contain some broken readings. This module applies the
"threshold + report" policy:

  - keep the readings that pass the SensorReading contract
  - quarantine the readings that fail, together with the REASON they failed
  - if too many readings fail (failure rate > threshold), reject the WHOLE batch
    and flag it, because that many failures signals a systemic problem
    (mis-configured sensor, broken feed) rather than a one-off glitch.

This keeps usable data flowing while making a degrading feed impossible to hide.
"""

from dataclasses import dataclass, field

from pydantic import ValidationError

from src.ingestion.contracts import SensorReading

# Max fraction of a batch allowed to fail before the WHOLE batch is rejected.
# Matches the project data contract: "no missing readings beyond threshold".
DEFAULT_FAILURE_THRESHOLD = 0.10  # 10%


@dataclass
class QuarantinedReading:
    """A reading that failed validation, kept with the reason why."""

    raw: dict
    reason: str


@dataclass
class BatchValidationResult:
    """The outcome of validating one micro-batch."""

    valid_readings: list[SensorReading] = field(default_factory=list)
    quarantined: list[QuarantinedReading] = field(default_factory=list)
    accepted: bool = True
    verdict: str = ""

    @property
    def total(self) -> int:
        return len(self.valid_readings) + len(self.quarantined)

    @property
    def valid_count(self) -> int:
        return len(self.valid_readings)

    @property
    def invalid_count(self) -> int:
        return len(self.quarantined)

    @property
    def failure_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.invalid_count / self.total


def validate_batch(
    raw_readings: list[dict],
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
) -> BatchValidationResult:
    """Validate a micro-batch of raw readings (threshold + report policy).

    Args:
        raw_readings: raw, unvalidated reading dicts as received from the feed.
        failure_threshold: max fraction allowed to fail before the batch is rejected.

    Returns:
        BatchValidationResult: clean readings, quarantined readings (with reasons),
        and a verdict on whether the batch as a whole is accepted.
    """
    result = BatchValidationResult()

    # Validate each reading by delegating to the SensorReading contract.
    for raw in raw_readings:
        try:
            result.valid_readings.append(SensorReading(**raw))
        except ValidationError as e:
            result.quarantined.append(QuarantinedReading(raw=raw, reason=str(e)))

    # An empty batch is itself a problem worth flagging (possible silent feed).
    if result.total == 0:
        result.accepted = False
        result.verdict = "REJECTED: empty batch - no readings received"
        return result

    # Reject the whole batch only if the failure rate exceeds the threshold.
    if result.failure_rate > failure_threshold:
        result.accepted = False
        result.verdict = (
            f"REJECTED: failure rate {result.failure_rate:.0%} exceeds threshold "
            f"{failure_threshold:.0%} ({result.invalid_count}/{result.total} failed) "
            f"- systemic data-quality problem, investigate the feed/sensors"
        )
    else:
        result.accepted = True
        result.verdict = (
            f"ACCEPTED: {result.valid_count}/{result.total} valid; "
            f"{result.invalid_count} quarantined "
            f"(failure rate {result.failure_rate:.0%}, within {failure_threshold:.0%})"
        )

    return result

"""Request/response schemas for the shared-FIR EQ trial API."""

from __future__ import annotations

import math
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator

MAX_IR_LENGTH = 256
MIN_SEATS = 2
MAX_SEATS = 4
MAX_TARGET_DELAY = 31
MIN_GAIN_LIMIT = 1.0
FIR_TAPS = 32


class OptimizeRequest(BaseModel):
    """Trial-optimisation request."""

    impulse_responses: list[list[float]] = Field(
        min_length=MIN_SEATS,
        max_length=MAX_SEATS,
        description=(
            "2-4 seat impulse responses. All share the same sample rate, are "
            "aligned on the direct-sound onset, have equal length, and contain "
            "at most 256 samples each."
        ),
    )
    target_delay: int = Field(
        ge=0,
        le=MAX_TARGET_DELAY,
        description="Target delay in samples (0-31) of the desired unit impulse.",
    )
    gain_limit: float = Field(
        ge=MIN_GAIN_LIMIT,
        description="Upper bound on the L1 norm (sum of absolute values) of the 32 FIR coefficients; must be >= 1.",
    )
    sample_rate: float | None = Field(
        default=None,
        gt=0,
        description="Optional shared sample rate in Hz (metadata only, echoed back).",
    )

    @field_validator("impulse_responses")
    @classmethod
    def _check_impulse_responses(cls, value: list[list[float]]) -> list[list[float]]:
        lengths = {len(ir) for ir in value}
        if 0 in lengths:
            raise ValueError("impulse responses must not be empty")
        if len(lengths) != 1:
            raise ValueError("all impulse responses must have the same length")
        if lengths.pop() > MAX_IR_LENGTH:
            raise ValueError(f"impulse responses are limited to {MAX_IR_LENGTH} samples each")
        for ir in value:
            for sample in ir:
                if not math.isfinite(sample):
                    raise ValueError("impulse responses must contain only finite values")
        return value


class SeatBeforeAfter(BaseModel):
    seat: int
    error_before: float = Field(description="MSE of the unit-gain passthrough at the same target delay.")
    error_after: float = Field(description="MSE with the returned FIR applied (full convolution).")


class SeatBaseline(BaseModel):
    seat: int
    error_before: float = Field(description="MSE of the unit-gain passthrough at the requested target delay.")


class ImprovedResponse(BaseModel):
    """Returned only when the worst-seat MSE was strictly reduced."""

    status: Literal["improved"] = "improved"
    detail: str
    taps: int
    target_delay: int
    gain_limit: float
    l1_norm: float = Field(description="Sum of absolute values of the returned coefficients (<= gain_limit).")
    coefficients: list[float] = Field(description="32 causal FIR coefficients to load into the loudspeaker EQ.")
    seats: list[SeatBeforeAfter]
    worst_error_before: float
    worst_error_after: float
    solver_success: bool


class NoImprovementResponse(BaseModel):
    """The optimisation ran successfully but no feasible FIR beats the passthrough."""

    status: Literal["no_improvement"] = "no_improvement"
    detail: str
    seats: list[SeatBaseline]
    worst_error_before: float


class FailedResponse(BaseModel):
    """The solver failed to produce a trustworthy answer."""

    status: Literal["failed"] = "failed"
    detail: str


OptimizeResponse = Annotated[
    Union[ImprovedResponse, NoImprovementResponse, FailedResponse],
    Field(discriminator="status"),
]

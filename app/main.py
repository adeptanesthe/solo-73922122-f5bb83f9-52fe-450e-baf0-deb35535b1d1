"""Theatre speaker EQ trial API.

A theatre loudspeaker can only load one set of EQ parameters, so this
service trial-computes a single 32-tap causal FIR shared by 2-4 seats.

Given the seats' equal-length impulse responses (same sample rate,
aligned on the direct-sound onset, <= 256 samples each), a target delay
(0-31 samples) and an L1 gain limit (>= 1), it minimises the worst
per-seat MSE between the full convolution ``h_k * g`` and a delayed unit
impulse, without making any seat worse than the plain passthrough.

Coefficients and per-seat before/after errors are returned only when the
worst error actually drops; ``no_improvement`` and ``failed`` outcomes
are reported separately.
"""

from __future__ import annotations

import math
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import solver
from app.schemas import (
    FailedResponse,
    ImprovedResponse,
    NoImprovementResponse,
    OptimizeRequest,
    OptimizeResponse,
    SeatBaseline,
    SeatBeforeAfter,
)

app = FastAPI(
    title="Theatre Speaker EQ Trial API",
    version="1.0.0",
    description=__doc__,
)

EXAMPLE_REQUEST = {
    "sample_rate": 48000,
    "target_delay": 1,
    "gain_limit": 2.0,
    "impulse_responses": [
        [1.0, 0.5],
        [1.0, -0.5],
    ],
}


def _make_json_safe(value: Any) -> Any:
    """Replace non-finite floats so json.dumps(allow_nan=False) never fails."""
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {key: _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Same shape as FastAPI's default 422 handler, but sanitised: Python's
    # JSON parser accepts NaN/Infinity, and echoing such rejected input back
    # verbatim would otherwise crash the error serialisation itself.
    content = jsonable_encoder({"detail": exc.errors()})
    return JSONResponse(status_code=422, content=_make_json_safe(content))


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "service": "theatre-speaker-eq-trial",
        "docs": "/docs",
        "health": "/health",
        "optimize": "POST /v1/eq/optimize",
        "example_request": "GET /v1/eq/example",
    }


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/eq/example", tags=["demo"])
def example() -> dict:
    return {
        "description": "POST this body to /v1/eq/optimize (two seats whose reflections need opposite EQ corrections).",
        "request": EXAMPLE_REQUEST,
    }


@app.post(
    "/v1/eq/optimize",
    response_model=OptimizeResponse,
    tags=["optimize"],
    summary="Trial-optimise one shared 32-tap causal FIR for 2-4 seats",
)
def optimize(request: OptimizeRequest) -> OptimizeResponse:
    result = solver.solve_shared_fir(
        impulse_responses=request.impulse_responses,
        target_delay=request.target_delay,
        gain_limit=request.gain_limit,
    )

    if result.status == solver.STATUS_IMPROVED:
        return ImprovedResponse(
            detail=result.detail,
            taps=solver.FIR_TAPS,
            target_delay=request.target_delay,
            gain_limit=request.gain_limit,
            l1_norm=result.l1_norm,
            coefficients=result.coefficients,
            seats=[
                SeatBeforeAfter(seat=i, error_before=before, error_after=after)
                for i, (before, after) in enumerate(zip(result.errors_before, result.errors_after))
            ],
            worst_error_before=result.worst_before,
            worst_error_after=result.worst_after,
            solver_success=bool(result.solver_success),
        )

    if result.status == solver.STATUS_NO_IMPROVEMENT:
        return NoImprovementResponse(
            detail=result.detail,
            seats=[
                SeatBaseline(seat=i, error_before=before)
                for i, before in enumerate(result.errors_before)
            ],
            worst_error_before=result.worst_before,
        )

    return FailedResponse(detail=result.detail)

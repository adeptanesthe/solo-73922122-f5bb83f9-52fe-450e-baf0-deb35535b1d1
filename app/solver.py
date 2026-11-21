"""Shared-FIR equalizer optimisation for a multi-seat listening area.

The theatre loudspeaker can only load one set of EQ coefficients, so a
single 32-tap causal FIR ``g`` is optimised for all seats at once.

For seat ``k`` with impulse response ``h_k`` (all seats share the same
sample rate and are aligned on the direct-sound onset) the equalised
response is the *full* convolution ``y_k = h_k * g`` of length
``N = len(h_k) + 31``.  The target is a unit impulse delayed by
``target_delay`` samples, and the per-seat error is the mean squared
error between ``y_k`` and that delayed impulse.

The optimiser minimises the worst per-seat MSE subject to

* ``sum(abs(g)) <= gain_limit`` (the hardware gain budget), and
* ``mse_k(g) <= mse_k(passthrough)`` for every seat, i.e. no seat may
  end up worse than the plain unit-gain passthrough at the same delay.

The problem is convex (a max of convex quadratics under convex
constraints) and is solved in epigraph form with SLSQP.  The baseline
passthrough FIR ``[1, 0, ...]`` is always feasible because
``gain_limit >= 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve

FIR_TAPS = 32

STATUS_IMPROVED = "improved"
STATUS_NO_IMPROVEMENT = "no_improvement"
STATUS_FAILED = "failed"


@dataclass
class SolveResult:
    status: str
    detail: str
    coefficients: list[float] | None
    l1_norm: float | None
    errors_before: list[float]
    errors_after: list[float] | None
    worst_before: float
    worst_after: float | None
    solver_success: bool | None


def _convolution_matrix(h: np.ndarray, taps: int) -> np.ndarray:
    """Dense matrix ``H`` such that ``H @ g == convolve(h, g, mode="full")``."""
    n_out = h.size + taps - 1
    mat = np.zeros((n_out, taps))
    for col in range(taps):
        mat[col:col + h.size, col] = h
    return mat


def full_length_mse(h, g, target_delay: int) -> float:
    """MSE between the full convolution ``h * g`` and the delayed unit impulse."""
    y = convolve(np.asarray(h, dtype=np.float64), np.asarray(g, dtype=np.float64), mode="full")
    target = np.zeros(y.size)
    target[target_delay] = 1.0
    residual = y - target
    return float(residual @ residual) / residual.size


def solve_shared_fir(
    impulse_responses: list[list[float]],
    target_delay: int,
    gain_limit: float,
    taps: int = FIR_TAPS,
) -> SolveResult:
    """Optimise one shared causal FIR for all seats.

    Returns a :class:`SolveResult` whose ``status`` is one of
    ``"improved"`` / ``"no_improvement"`` / ``"failed"``.  Coefficients
    and per-seat before/after errors are only populated when the worst
    seat error was strictly reduced.
    """
    hs = [np.asarray(ir, dtype=np.float64) for ir in impulse_responses]
    n_seats = len(hs)
    n_out = hs[0].size + taps - 1

    mats = [_convolution_matrix(h, taps) for h in hs]
    target = np.zeros(n_out)
    target[target_delay] = 1.0
    scale = 1.0 / n_out

    def seat_errors(g: np.ndarray) -> np.ndarray:
        return np.array([scale * float(np.sum((m @ g - target) ** 2)) for m in mats])

    baseline = np.zeros(taps)
    baseline[0] = 1.0
    errors_before = seat_errors(baseline)
    worst_before = float(errors_before.max())

    def _result(
        status: str,
        detail: str,
        coefficients: np.ndarray | None = None,
        l1_norm: float | None = None,
        errors_after: np.ndarray | None = None,
        worst_after: float | None = None,
        solver_success: bool | None = None,
    ) -> SolveResult:
        return SolveResult(
            status=status,
            detail=detail,
            coefficients=None if coefficients is None else [float(c) for c in coefficients],
            l1_norm=l1_norm,
            errors_before=[float(e) for e in errors_before],
            errors_after=None if errors_after is None else [float(e) for e in errors_after],
            worst_before=worst_before,
            worst_after=worst_after,
            solver_success=solver_success,
        )

    if worst_before <= 1e-15:
        return _result(
            STATUS_NO_IMPROVEMENT,
            "passthrough already matches the delayed unit impulse (worst MSE ~ 0); nothing to improve",
        )

    # ----- convex epigraph formulation -------------------------------------
    # variables x = [g (taps), t];  minimise t  subject to
    #   e_k(g) <= t                 for every seat k   (minimax objective)
    #   e_k(g) <= e_k(baseline)     for every seat k   (no seat gets worse)
    #   sum(abs(g)) <= gain_limit                      (hardware gain budget)

    def objective(x: np.ndarray) -> float:
        return float(x[taps])

    def objective_jac(x: np.ndarray) -> np.ndarray:
        grad = np.zeros(taps + 1)
        grad[taps] = 1.0
        return grad

    def constraints(x: np.ndarray) -> np.ndarray:
        g = x[:taps]
        t = x[taps]
        errs = seat_errors(g)
        return np.concatenate((
            t - errs,
            errors_before - errs,
            [gain_limit - float(np.abs(g).sum())],
        ))

    def constraints_jac(x: np.ndarray) -> np.ndarray:
        g = x[:taps]
        jac = np.zeros((2 * n_seats + 1, taps + 1))
        for i, m in enumerate(mats):
            grad_e = 2.0 * scale * (m.T @ (m @ g - target))
            jac[i, :taps] = -grad_e
            jac[i, taps] = 1.0
            jac[n_seats + i, :taps] = -grad_e
        jac[2 * n_seats, :taps] = -np.sign(g)  # subgradient of the L1 norm
        return jac

    x0 = np.concatenate((baseline, [worst_before]))
    bounds = [(None, None)] * taps + [(0.0, None)]
    try:
        res = minimize(
            objective,
            x0,
            jac=objective_jac,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": constraints, "jac": constraints_jac}],
            options={"maxiter": 500, "ftol": 1e-12},
        )
    except Exception as exc:  # pragma: no cover - exercised via monkeypatched test
        return _result(STATUS_FAILED, f"optimizer raised an exception: {exc!r}")

    solver_success = bool(res.success)
    if res.x is None or not np.all(np.isfinite(res.x)):
        return _result(
            STATUS_FAILED,
            f"optimizer returned a non-finite iterate ({res.message})",
            solver_success=solver_success,
        )

    g = np.array(res.x[:taps], dtype=np.float64)
    l1 = float(np.abs(g).sum())
    if l1 > gain_limit:
        # Trim any tiny SLSQP constraint overshoot, then re-evaluate exactly.
        g *= gain_limit / l1
        l1 = float(np.abs(g).sum())

    errors_after = seat_errors(g)
    worst_after = float(errors_after.max())

    feas_tol = 1e-9 * np.maximum(1.0, errors_before)
    feasible = bool(np.all(errors_after <= errors_before + feas_tol))
    improvement_tol = 1e-9 * max(1.0, worst_before)

    if feasible and worst_after < worst_before - improvement_tol:
        detail = "worst-seat MSE reduced vs. passthrough"
        if not solver_success:
            detail += f" (solver stopped early: {res.message}; returned point verified feasible)"
        return _result(
            STATUS_IMPROVED,
            detail,
            coefficients=g,
            l1_norm=l1,
            errors_after=errors_after,
            worst_after=worst_after,
            solver_success=solver_success,
        )
    if not solver_success:
        return _result(
            STATUS_FAILED,
            f"optimizer did not converge: {res.message}",
            solver_success=False,
        )
    if not feasible:
        return _result(
            STATUS_FAILED,
            "optimizer converged but the iterate violates the per-seat non-regression constraints",
            solver_success=True,
        )
    return _result(
        STATUS_NO_IMPROVEMENT,
        "no feasible 32-tap FIR within the gain limit lowers the worst-seat MSE",
        solver_success=True,
    )

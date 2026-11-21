"""End-to-end and unit tests for the shared-FIR EQ trial API."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import solver
from app.main import app

client = TestClient(app)

H1 = [1.0, 0.5]
H2 = [1.0, -0.5]


def make_payload(**overrides) -> dict:
    payload = {
        "impulse_responses": [H1, H2],
        "target_delay": 1,
        "gain_limit": 2.0,
    }
    payload.update(overrides)
    return payload


def reference_errors(impulse_responses, coefficients, target_delay) -> list[float]:
    """Independent recomputation of the per-seat full-convolution MSE."""
    errors = []
    for h in impulse_responses:
        y = np.convolve(np.asarray(h, float), np.asarray(coefficients, float))
        target = np.zeros(y.size)
        target[target_delay] = 1.0
        errors.append(float(np.mean((y - target) ** 2)))
    return errors


# ---------------------------------------------------------------- meta


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_example_endpoint_returns_postable_body():
    resp = client.get("/v1/eq/example")
    assert resp.status_code == 200
    body = resp.json()["request"]
    result = client.post("/v1/eq/optimize", json=body)
    assert result.status_code == 200
    assert result.json()["status"] == "improved"


# ------------------------------------------------------- improved path


def test_improved_two_seats():
    resp = client.post("/v1/eq/optimize", json=make_payload())
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "improved"
    assert len(body["coefficients"]) == 32
    assert body["l1_norm"] == pytest.approx(sum(abs(c) for c in body["coefficients"]), rel=1e-12)
    assert body["l1_norm"] <= 2.0 + 1e-9

    seats = body["seats"]
    assert len(seats) == 2
    for seat in seats:
        assert seat["error_after"] <= seat["error_before"] + 1e-12
    assert body["worst_error_after"] < body["worst_error_before"]
    assert body["worst_error_before"] == pytest.approx(max(s["error_before"] for s in seats))
    assert body["worst_error_after"] == pytest.approx(max(s["error_after"] for s in seats))

    # Cross-check the reported numbers against an independent recomputation.
    after = reference_errors([H1, H2], body["coefficients"], target_delay=1)
    before = reference_errors([H1, H2], [1.0] + [0.0] * 31, target_delay=1)
    for seat, exp_before, exp_after in zip(seats, before, after):
        assert seat["error_before"] == pytest.approx(exp_before, rel=1e-9)
        assert seat["error_after"] == pytest.approx(exp_after, rel=1e-9)


def test_gain_limit_one_still_improves_and_respects_budget():
    resp = client.post("/v1/eq/optimize", json=make_payload(gain_limit=1.0))
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "improved"
    assert body["l1_norm"] <= 1.0 + 1e-9
    assert body["worst_error_after"] < body["worst_error_before"]


def test_identical_seats_recover_inverse_filter():
    # Both seats equal: the optimum is the (delayed, truncated) inverse of
    # [1, 0.5], which drives the MSE close to zero within the gain budget.
    resp = client.post(
        "/v1/eq/optimize",
        json=make_payload(impulse_responses=[H1, H1], gain_limit=2.0),
    )
    body = resp.json()
    assert body["status"] == "improved"
    assert body["worst_error_after"] < 1e-6
    assert body["seats"][0]["error_after"] == pytest.approx(body["seats"][1]["error_after"], rel=1e-6)


def test_max_delay_and_four_seats_and_max_length():
    rng = np.random.default_rng(20261121)
    irs = (0.5 * rng.standard_normal((4, 256))).tolist()
    irs[0][0] = 1.0  # make sure there is a direct sound
    resp = client.post(
        "/v1/eq/optimize",
        json={"impulse_responses": irs, "target_delay": 31, "gain_limit": 3.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"improved", "no_improvement"}
    if body["status"] == "improved":
        assert len(body["coefficients"]) == 32
        assert len(body["seats"]) == 4
        assert body["l1_norm"] <= 3.0 + 1e-9
        for seat in body["seats"]:
            assert seat["error_after"] <= seat["error_before"] + 1e-9


# --------------------------------------------------- no_improvement path


def test_no_improvement_when_baseline_is_already_perfect():
    resp = client.post(
        "/v1/eq/optimize",
        json=make_payload(impulse_responses=[[0.0, 1.0], [0.0, 1.0]], target_delay=1),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "no_improvement"
    assert "coefficients" not in body
    assert body["worst_error_before"] == pytest.approx(0.0, abs=1e-15)
    assert len(body["seats"]) == 2


def test_no_improvement_with_zero_impulse_responses():
    # y_k is identically zero for every FIR, so the MSE cannot drop.
    resp = client.post(
        "/v1/eq/optimize",
        json=make_payload(impulse_responses=[[0.0, 0.0], [0.0, 0.0]]),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "no_improvement"
    assert "coefficients" not in body


# ---------------------------------------------------------- failed path


def test_solver_exception_is_reported_as_failed(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(solver, "minimize", boom)
    resp = client.post("/v1/eq/optimize", json=make_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert "coefficients" not in body
    assert "boom" in body["detail"]


def test_solver_non_convergence_is_reported_as_failed(monkeypatch):
    class FakeResult:
        success = False
        message = "Iteration limit reached"
        # Passthrough iterate: feasible but not improving.
        x = np.concatenate(([1.0], np.zeros(32)))

    monkeypatch.setattr(solver, "minimize", lambda *a, **k: FakeResult())
    resp = client.post("/v1/eq/optimize", json=make_payload())
    body = resp.json()
    # The iterate does not improve and the solver did not converge, so the
    # outcome must be reported as a failure.
    assert body["status"] == "failed"
    assert "coefficients" not in body


# ------------------------------------------------------------ validation


@pytest.mark.parametrize(
    "override",
    [
        {"impulse_responses": [H1]},                                  # too few seats
        {"impulse_responses": [H1, H2, H1, H2, H1]},                  # too many seats
        {"impulse_responses": [[1.0, 0.5], [1.0, 0.5, 0.25]]},        # unequal lengths
        {"impulse_responses": [[], [1.0]]},                           # empty IR
        {"impulse_responses": [[0.1] * 257, [0.1] * 257]},            # too long
        {"target_delay": -1},
        {"target_delay": 32},
        {"target_delay": 1.5},
        {"gain_limit": 0.99},
        {"gain_limit": 0.0},
        {"sample_rate": 0.0},
    ],
)
def test_invalid_requests_return_422(override):
    resp = client.post("/v1/eq/optimize", json=make_payload(**override))
    assert resp.status_code == 422


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_samples_return_422(token):
    # Strict JSON has no NaN/Infinity, but Python's json parser accepts
    # these tokens, so the API must reject them explicitly.
    raw = (
        '{"impulse_responses": [[1.0, ' + token + '], [1.0, -0.5]],'
        ' "target_delay": 1, "gain_limit": 2.0}'
    )
    resp = client.post(
        "/v1/eq/optimize",
        content=raw,
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422


def test_missing_fields_return_422():
    assert client.post("/v1/eq/optimize", json={}).status_code == 422
    assert client.post(
        "/v1/eq/optimize", json={"impulse_responses": [H1, H2]}
    ).status_code == 422


# ------------------------------------------------------- solver unit API


def test_solver_direct_call_improved():
    result = solver.solve_shared_fir([H1, H2], target_delay=1, gain_limit=2.0)
    assert result.status == solver.STATUS_IMPROVED
    assert len(result.coefficients) == 32
    assert result.l1_norm <= 2.0 + 1e-9
    assert result.worst_after < result.worst_before
    for before, after in zip(result.errors_before, result.errors_after):
        assert after <= before + 1e-12


def test_full_length_mse_helper():
    # h * [1] == h; with delay 1 and h = [0, 1] the match is perfect.
    assert solver.full_length_mse([0.0, 1.0], [1.0], 1) == pytest.approx(0.0)
    # Passthrough of h = [1, 0.5] against a unit impulse at delay 1:
    # y = [1, 0.5] (N = 2), target = [0, 1], residual = [1, -0.5].
    assert solver.full_length_mse(H1, [1.0], 1) == pytest.approx((1.0 + 0.25) / 2)

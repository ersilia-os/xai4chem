"""The soft/hard merge: exact additivity, the rescale rule under a decreasing map, and the gate."""

import numpy as np
import pytest

from xai4chem.models.hard import A_CEILING, A_MIN, apply_isotonic, blend_ceiling, ramp


def test_rescale_rule_is_exact_for_a_decreasing_map():
    rng = np.random.default_rng(0)
    iso = {"x": [-3.0, -1.0, 0.0, 2.0, 4.0], "y": [2.0, 1.5, 0.0, -1.0, -2.5]}
    phi_H = rng.normal(size=(50, 7))
    base_H = np.full(50, 0.3)
    h = phi_H.sum(1) + base_H
    HS, base_HS = apply_isotonic(h, iso), apply_isotonic(base_H, iso)
    slope = (HS - base_HS) / (h - base_H)
    phi_HS = phi_H * slope[:, None]
    np.testing.assert_allclose(phi_HS.sum(1) + base_HS, HS, atol=1e-10)
    assert (slope <= 0).all()  # a decreasing map flips the signs


def test_gate_and_ceiling():
    assert ramp(0.2) == 0 and ramp(0.7) == 1 and 0 < ramp(0.55) < 1
    assert blend_ceiling(0.95) == A_CEILING == 0.4
    assert blend_ceiling(0.15, "regression") == pytest.approx(0.15)  # R² still binds below the ceiling
    assert blend_ceiling(0.95, "regression") == 0.25
    assert blend_ceiling(A_MIN / 2) == 0.0
    assert blend_ceiling(float("nan")) == 0.0
    assert blend_ceiling(-0.3) == 0.0

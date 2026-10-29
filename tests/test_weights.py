"""Target reweighting for S (Olinda's rule) and the hard-label task choice."""

import numpy as np
import pytest

from xai4chem.models import weights as W
from xai4chem.models.hard import detect_task, resolve_task


def test_balanced_target_is_not_weighted():
    y = np.random.default_rng(0).uniform(0, 1, 5000)
    assert W.choose_strategy(y)[0] == "none"
    assert W.resolve_weights(y)[:2] == (None, None)


def test_skewed_continuous_target_gets_density_weights():
    y = np.random.default_rng(0).exponential(1.0, 5000)
    edges, weights, info = W.resolve_weights(y)
    assert info["strategy"] == "kde"
    assert weights.min() >= 1.0 and weights.max() <= W.MAX_WEIGHT
    w = W.apply_weights(y, edges, weights)
    assert w[y > np.percentile(y, 99)].mean() > w[y < np.median(y)].mean()  # the tail counts for more


def test_skewed_discrete_or_small_target_gets_bins():
    y = np.r_[np.zeros(950), np.ones(50)]
    assert W.choose_strategy(y)[0] == "bins"
    small = np.random.default_rng(1).exponential(1.0, 200)
    assert W.choose_strategy(small)[0] == "bins"


def test_constant_target_and_modes():
    y = np.full(500, 0.3)
    assert W.resolve_weights(y)[2]["strategy"] == "none"
    assert W.choose_strategy(np.random.default_rng(0).uniform(0, 1, 5000), "on")[0] == "kde"
    assert W.choose_strategy(np.random.default_rng(0).exponential(1, 5000), "off")[0] == "none"
    with pytest.raises(ValueError, match="Unknown weighting mode"):
        W.choose_strategy(y, "sometimes")


def test_task_is_detected_or_forced():
    binary, scores = np.array([0.0, 1, 1, 0]), np.array([0.2, 0.9, 1.0, 0.0])
    assert detect_task(binary) == "binary" and detect_task(scores) == "regression"
    assert resolve_task(scores) == "regression"
    assert resolve_task(binary, "regression") == "regression"  # allowed: a classifier's labels as numbers
    with pytest.raises(ValueError, match="forced to --task binary"):
        resolve_task(scores, "binary", "hard column 'x'")
    with pytest.raises(ValueError, match="Unknown task"):
        resolve_task(binary, "multiclass")


def test_fidelity_reports_the_tail_separately():
    from xai4chem.models.surrogate import fidelity

    y = np.r_[np.zeros(90), np.arange(1.0, 11.0)]
    pred = y.copy()
    pred[-10:] += 2.0  # the bulk is perfect, the top decile is off by 2
    f = fidelity(y, pred)
    assert f["top_decile_rmse"] == pytest.approx(2.0)
    assert f["rmse"] < f["top_decile_rmse"]

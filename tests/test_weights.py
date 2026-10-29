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


def test_validation_split_is_value_stratified():
    from xai4chem.models.surrogate import split_indices

    y = np.random.default_rng(0).exponential(1.0, 1000)
    trn, val = split_indices(y, 0.1)
    assert len(val) == 100 and len(trn) == 900 and not set(trn) & set(val)
    assert np.allclose(np.quantile(y[val], [0.1, 0.5, 0.9]), np.quantile(y, [0.1, 0.5, 0.9]), rtol=0.1)
    assert y[val].min() == y.min() and y[val].max() == y.max()  # both extremes are validated


def test_target_checks_follow_olinda():
    from xai4chem.models.surrogate import check_target

    assert check_target("a", np.random.default_rng(0).normal(size=2000)) == []
    assert "noisy" in check_target("a", np.random.default_rng(0).normal(size=500))[0]
    assert "non-finite" in check_target("a", np.r_[np.random.default_rng(0).normal(size=2000), np.nan])[0]
    with pytest.raises(ValueError, match="at least 100"):
        check_target("a", np.arange(50.0))
    with pytest.raises(ValueError, match="constant"):
        check_target("a", np.full(500, 2.0))


def test_hard_label_limits_follow_olinda():
    from xai4chem.models.hard import check_hard_labels

    assert check_hard_labels("h", np.array([0.0, 0, 1, 1]), "binary") == []  # 4 rows, 2 of each, is enough
    with pytest.raises(ValueError, match="usable row"):
        check_hard_labels("h", np.array([0.0, 1, 1]), "binary")
    with pytest.raises(ValueError, match="1 positive and 5 negative"):
        check_hard_labels("h", np.array([0.0] * 5 + [1.0]), "binary")
    with pytest.raises(ValueError, match="at least 20"):
        check_hard_labels("h", np.linspace(0, 1, 10), "regression")
    with pytest.raises(ValueError, match="single value"):
        check_hard_labels("h", np.full(30, 0.7), "regression")
    assert check_hard_labels("h", np.linspace(0, 1, 30), "regression") == []


def test_label_scale_is_a_positive_magnitude_for_any_labels():
    """Olinda's rule: the mean absolute value, or 1 when there is nothing to divide by."""
    from xai4chem.models.hard import label_scale

    assert label_scale(np.array([2e6, 4e6])) == pytest.approx(3e6)
    assert label_scale(np.array([-3.0, 3.0])) == pytest.approx(3.0)
    assert label_scale(np.array([0.0, 0.0])) == 1.0
    assert label_scale(np.array([np.nan])) == 1.0

"""Tests for the diagnostics.

The chi-squared quantiles are checked against published tables, since
they are computed here rather than imported.  The calibration measures
are then checked in both directions: a correctly specified filter must
pass them, and a filter given the wrong Q must fail them -- a diagnostic
that never fires is no diagnostic at all.
"""

import numpy as np
import pytest

from lgssm import (
    LGSSM,
    chi2_cdf,
    chi2_quantile,
    consistency_interval,
    coverage,
    joint_coverage,
    kalman_filter,
    monte_carlo_nees,
    nees,
    nis,
    rmse,
    simulate,
)


@pytest.fixture
def walk():
    """A two-dimensional random walk, observed in both components."""
    return LGSSM(
        F=np.eye(2),
        Q=np.diag([0.5, 0.2]),
        H=np.eye(2),
        R=np.diag([0.8, 0.3]),
        m0=np.zeros(2),
        P0=np.eye(2),
    )


# ---------------------------------------------------------------------------
# the chi-squared distribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dof, p, expected",
    [
        (1, 0.95, 3.8415),
        (1, 0.975, 5.0239),
        (2, 0.95, 5.9915),
        (3, 0.05, 0.3518),
        (10, 0.05, 3.9403),
        (10, 0.95, 18.3070),
        (100, 0.95, 124.3421),
    ],
)
def test_chi2_quantile_matches_published_tables(dof, p, expected):
    assert chi2_quantile(p, dof) == pytest.approx(expected, abs=5e-4)


@pytest.mark.parametrize("dof", [1, 2, 7, 50])
@pytest.mark.parametrize("p", [0.01, 0.25, 0.5, 0.9, 0.99])
def test_chi2_cdf_inverts_the_quantile(dof, p):
    assert chi2_cdf(chi2_quantile(p, dof), dof) == pytest.approx(p, abs=1e-10)


def test_chi2_cdf_is_zero_below_the_support():
    assert chi2_cdf(-1.0, 3) == 0.0
    assert chi2_cdf(0.0, 3) == 0.0


@pytest.mark.parametrize("p", [0.0, 1.0, -0.5, 2.0])
def test_chi2_quantile_rejects_impossible_probabilities(p):
    with pytest.raises(ValueError, match="strictly between"):
        chi2_quantile(p, 2)


def test_chi2_rejects_bad_dof():
    with pytest.raises(ValueError, match="dof"):
        chi2_quantile(0.5, 0)


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------


def test_rmse_is_the_root_mean_square_of_the_errors():
    estimates = np.array([[0.0, 0.0], [1.0, 1.0]])
    truth = np.array([[3.0, 0.0], [1.0, 5.0]])
    # errors 3, 0, 0, 4 -> mean square 25/4
    assert rmse(estimates, truth) == pytest.approx(2.5)
    np.testing.assert_allclose(
        rmse(estimates, truth, per_dimension=True), [3.0 / np.sqrt(2), 4.0 / np.sqrt(2)]
    )


def test_rmse_is_zero_for_a_perfect_estimate(rng):
    truth = rng.standard_normal((10, 3))
    assert rmse(truth, truth) == 0.0


def test_off_by_one_truth_gets_a_useful_error(walk, rng):
    """The likeliest mistake: passing states, not states[1:]."""
    states, obs = simulate(walk, 20, rng)
    result = kalman_filter(walk, obs)
    with pytest.raises(ValueError, match=r"pass states\[1:\]"):
        rmse(result.means, states)


def test_mismatched_shapes_are_rejected(rng):
    with pytest.raises(ValueError, match="estimates have shape"):
        rmse(rng.standard_normal((10, 2)), rng.standard_normal((10, 3)))


def test_smoothing_lowers_rmse(walk, rng):
    from lgssm import rts_smoother

    states, obs = simulate(walk, 400, rng)
    filtered = kalman_filter(walk, obs)
    smoothed = rts_smoother(walk, filtered)
    assert rmse(smoothed.means, states[1:]) < rmse(filtered.means, states[1:])


# ---------------------------------------------------------------------------
# NEES
# ---------------------------------------------------------------------------


def test_nees_is_the_squared_error_in_covariance_units():
    means = np.array([[0.0], [1.0]])
    covs = np.array([[[4.0]], [[0.25]]])
    truth = np.array([[2.0], [2.0]])
    np.testing.assert_allclose(nees(means, covs, truth), [1.0, 4.0])


def test_nees_averages_the_state_dimension_when_the_model_is_right(walk):
    values = monte_carlo_nees(walk, n_steps=20, n_runs=200, rng=np.random.default_rng(0))
    assert values.shape == (200, 20)
    # Samples are independent across runs but correlated down a run, so the
    # standard error of this average is nearer 0.1 than the 0.03 that 4000
    # independent draws would give.  See the column-wise test below for the
    # version of this check that is exact.
    assert values.mean() == pytest.approx(walk.dim_x, rel=0.1)


def test_nees_accepts_a_correct_filter(walk):
    """The Bar-Shalom test, run the way it is exact: across runs."""
    n_runs = 200
    values = monte_carlo_nees(walk, n_steps=20, n_runs=n_runs, rng=np.random.default_rng(1))
    low, high = consistency_interval(walk.dim_x, n_runs)
    inside = [(low <= column.mean() <= high) for column in values.T]
    assert sum(inside) >= 17  # 20 columns, each a 95% test


def test_nees_rejects_an_overconfident_filter(walk):
    """Filtering with Q far too small: the estimates are not wild, but the
    error bars are, and NEES is what notices."""
    overconfident = LGSSM(
        F=walk.F, Q=walk.Q / 100.0, H=walk.H, R=walk.R, m0=walk.m0, P0=walk.P0
    )
    n_runs = 100
    values = monte_carlo_nees(
        walk, n_steps=20, n_runs=n_runs, filter_model=overconfident,
        rng=np.random.default_rng(2),
    )
    _, high = consistency_interval(walk.dim_x, n_runs)
    assert values.mean() > high


def test_smoothed_estimates_are_calibrated_too(walk):
    values = monte_carlo_nees(
        walk, n_steps=20, n_runs=200, rng=np.random.default_rng(3), smooth=True
    )
    assert values.mean() == pytest.approx(walk.dim_x, rel=0.1)


def test_consistency_interval_brackets_the_dimension_and_tightens(walk):
    wide = consistency_interval(2, 10)
    narrow = consistency_interval(2, 1000)
    for low, high in (wide, narrow):
        assert low < 2 < high
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_consistency_interval_rejects_an_empty_sample():
    with pytest.raises(ValueError, match="n_samples"):
        consistency_interval(2, 0)


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------


def test_coverage_is_the_nominal_level_when_the_model_is_right(walk):
    states, obs = simulate(walk, 4000, np.random.default_rng(4))
    result = kalman_filter(walk, obs)
    covered = coverage(result.means, result.covs, states[1:])
    assert covered.shape == (2,)
    np.testing.assert_allclose(covered, 0.95, atol=0.02)


def test_coverage_falls_when_the_filter_is_overconfident(walk):
    states, obs = simulate(walk, 2000, np.random.default_rng(5))
    overconfident = LGSSM(
        F=walk.F, Q=walk.Q / 100.0, H=walk.H, R=walk.R, m0=walk.m0, P0=walk.P0
    )
    result = kalman_filter(overconfident, obs)
    assert np.all(coverage(result.means, result.covs, states[1:]) < 0.8)


def test_coverage_level_is_respected(walk):
    states, obs = simulate(walk, 4000, np.random.default_rng(6))
    result = kalman_filter(walk, obs)
    at_50 = coverage(result.means, result.covs, states[1:], level=0.5)
    np.testing.assert_allclose(at_50, 0.5, atol=0.03)


def test_joint_coverage_uses_the_whole_covariance(walk):
    states, obs = simulate(walk, 4000, np.random.default_rng(7))
    result = kalman_filter(walk, obs)
    assert joint_coverage(result.means, result.covs, states[1:]) == pytest.approx(
        0.95, abs=0.02
    )


# ---------------------------------------------------------------------------
# NIS, the one that works without a true state
# ---------------------------------------------------------------------------


def test_nis_averages_the_measurement_dimension(walk):
    _, obs = simulate(walk, 4000, np.random.default_rng(8))
    values, dofs = nis(kalman_filter(walk, obs))
    assert np.all(dofs == walk.dim_y)
    assert values.mean() == pytest.approx(walk.dim_y, rel=0.05)


def test_nis_rejects_a_misspecified_filter(walk):
    _, obs = simulate(walk, 2000, np.random.default_rng(9))
    wrong = LGSSM(
        F=walk.F, Q=walk.Q, H=walk.H, R=walk.R / 50.0, m0=walk.m0, P0=walk.P0
    )
    values, dofs = nis(kalman_filter(wrong, obs))
    _, high = consistency_interval(walk.dim_y, len(values))
    assert values.mean() > high


def test_nis_counts_degrees_of_freedom_through_gaps(walk):
    _, obs = simulate(walk, 50, np.random.default_rng(10))
    obs[3, 0] = np.nan
    obs[7, :] = np.nan

    values, dofs = nis(kalman_filter(walk, obs))
    assert dofs[3] == 1
    assert dofs[7] == 0
    assert values[7] == 0.0
    assert dofs.sum() == 50 * 2 - 3

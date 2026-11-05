"""Tests for the EM algorithm.

Two kinds of check here.  The structural one is that the observed-data
log-likelihood never decreases: that is what EM guarantees, and it holds
whatever the data, so it is asserted on every fit.  The sharp ones are the
M-step updates, which are checked against ordinary least squares on a model
rigged so that the states are effectively known -- when the posterior over
x collapses onto the truth, the imputed MLE must become the plain MLE.
"""

import numpy as np
import pytest

from lgssm import LGSSM, fit_em, kalman_filter, rts_smoother, simulate


def scalar_model(F=0.9, Q=0.5, R=0.8, P0=1.0):
    return LGSSM(
        F=[[F]], Q=[[Q]], H=[[1.0]], R=[[R]], m0=[0.0], P0=[[P0]]
    )


# ---------------------------------------------------------------------------
# the guarantee
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "estimate",
    [("Q",), ("R",), ("Q", "R"), ("F", "Q"), ("m0", "P0"), ("F", "Q", "H", "R")],
)
def test_loglik_never_decreases(rng, make_model, estimate):
    """The defining property of EM, asserted rather than assumed."""
    truth = make_model(rng)
    _, obs = simulate(truth, 300, rng)
    start = make_model(rng)

    result = fit_em(start, obs, estimate=estimate, max_iter=25)
    assert np.all(np.diff(result.loglik) >= -1e-8)


def test_fit_improves_a_bad_starting_point(rng):
    truth = scalar_model(Q=0.5, R=0.8)
    _, obs = simulate(truth, 500, rng)
    result = fit_em(scalar_model(Q=5.0, R=0.05), obs, max_iter=50)
    assert result.loglik_increase > 0


def test_final_loglik_belongs_to_the_returned_model(rng):
    """loglik[-1] must score result.model, converged or not."""
    truth = scalar_model()
    _, obs = simulate(truth, 200, rng)
    for max_iter in (3, 200):
        result = fit_em(scalar_model(Q=2.0, R=2.0), obs, max_iter=max_iter)
        direct = kalman_filter(result.model, obs).loglik
        assert result.loglik[-1] == pytest.approx(direct, abs=1e-9)


# ---------------------------------------------------------------------------
# recovery
# ---------------------------------------------------------------------------


def test_recovers_the_noise_covariances():
    """A long series, a deliberately wrong start, and known F and H."""
    rng = np.random.default_rng(0)
    truth = scalar_model(Q=0.5, R=0.8)
    _, obs = simulate(truth, 1500, rng)

    result = fit_em(scalar_model(Q=2.0, R=2.0), obs, max_iter=300, tol=1e-8)
    assert result.converged
    assert result.model.Q[0, 0] == pytest.approx(0.5, rel=0.2)
    assert result.model.R[0, 0] == pytest.approx(0.8, rel=0.2)


def test_recovers_the_transition_matrix():
    rng = np.random.default_rng(1)
    truth = scalar_model(F=0.7, Q=0.3, R=0.2)
    _, obs = simulate(truth, 2000, rng)

    start = scalar_model(F=0.1, Q=0.3, R=0.2)
    result = fit_em(start, obs, estimate=("F",), max_iter=300, tol=1e-8)
    assert result.model.F[0, 0] == pytest.approx(0.7, abs=0.05)


# ---------------------------------------------------------------------------
# the M-step, against least squares on known states
# ---------------------------------------------------------------------------


def test_transition_update_is_least_squares_when_states_are_known():
    """With almost no measurement noise the smoother returns the truth,
    so the F update must reduce to regressing x_t on x_{t-1}."""
    rng = np.random.default_rng(2)
    model = LGSSM(
        F=[[0.6]], Q=[[0.4]], H=[[1.0]], R=[[1e-8]], m0=[0.0], P0=[[1e-8]]
    )
    states, obs = simulate(model, 400, rng)

    one_step = fit_em(model, obs, estimate=("F",), max_iter=1)
    x, x_prev = states[1:, 0], states[:-1, 0]
    assert one_step.model.F[0, 0] == pytest.approx(
        (x * x_prev).sum() / (x_prev**2).sum(), rel=1e-4
    )


def test_process_noise_update_is_the_residual_variance_when_states_are_known():
    rng = np.random.default_rng(3)
    model = LGSSM(
        F=[[0.6]], Q=[[2.0]], H=[[1.0]], R=[[1e-8]], m0=[0.0], P0=[[1e-8]]
    )
    states, obs = simulate(model, 400, rng)

    one_step = fit_em(model, obs, estimate=("Q",), max_iter=1)
    residuals = states[1:, 0] - 0.6 * states[:-1, 0]
    assert one_step.model.Q[0, 0] == pytest.approx(np.mean(residuals**2), rel=1e-4)


def test_measurement_noise_update_is_exact_for_a_deterministic_state():
    """Q = 0 and P0 = 0 make the state known outright, so the R update is
    just the mean squared measurement, with no posterior correction."""
    rng = np.random.default_rng(4)
    model = LGSSM(
        F=np.ones((1, 1)), Q=np.zeros((1, 1)), H=np.ones((1, 1)),
        R=[[0.4]], m0=[0.0], P0=np.zeros((1, 1)),
    )
    _, obs = simulate(model, 500, rng)

    one_step = fit_em(model, obs, estimate=("R",), max_iter=1)
    assert one_step.model.R[0, 0] == pytest.approx(np.mean(obs[:, 0] ** 2), rel=1e-8)


def test_initial_state_updates_to_the_smoothed_initial_state(rng, make_model):
    truth = make_model(rng)
    _, obs = simulate(truth, 50, rng)
    start = make_model(rng)

    one_step = fit_em(start, obs, estimate=("m0", "P0"), max_iter=1)
    smoothed = rts_smoother(start, kalman_filter(start, obs))
    np.testing.assert_allclose(one_step.model.m0, smoothed.initial_mean, atol=1e-10)
    np.testing.assert_allclose(one_step.model.P0, smoothed.initial_cov, atol=1e-10)


def test_fixed_m0_shifts_the_p0_update():
    """With m0 held away from the smoothed initial state, the P0 update
    must pick up the squared offset as well as the smoothed covariance."""
    rng = np.random.default_rng(5)
    truth = scalar_model()
    _, obs = simulate(truth, 30, rng)
    start = scalar_model(P0=2.0)

    smoothed = rts_smoother(start, kalman_filter(start, obs))
    offset = smoothed.initial_mean - start.m0
    one_step = fit_em(start, obs, estimate=("P0",), max_iter=1)
    assert one_step.model.P0[0, 0] == pytest.approx(
        smoothed.initial_cov[0, 0] + offset[0] ** 2
    )


# ---------------------------------------------------------------------------
# what is held fixed
# ---------------------------------------------------------------------------


def test_nothing_to_estimate_leaves_the_model_alone(rng):
    truth = scalar_model()
    _, obs = simulate(truth, 100, rng)
    result = fit_em(truth, obs, estimate=(), max_iter=10)
    np.testing.assert_allclose(result.model.Q, truth.Q)
    np.testing.assert_allclose(result.model.R, truth.R)
    assert result.converged
    assert result.loglik[0] == pytest.approx(result.loglik[-1])


def test_unestimated_parameters_are_untouched(rng):
    truth = scalar_model()
    _, obs = simulate(truth, 100, rng)
    start = scalar_model(F=0.3, Q=2.0, R=2.0)
    result = fit_em(start, obs, estimate=("Q",), max_iter=20)

    np.testing.assert_allclose(result.model.F, start.F)
    np.testing.assert_allclose(result.model.R, start.R)
    assert not np.allclose(result.model.Q, start.Q)


def test_time_varying_h_may_be_known_while_the_noise_is_estimated():
    """The drifting-regression case: H_t carries the regressor, so it is
    known and time-varying, while Q and R are what we want to learn."""
    rng = np.random.default_rng(6)
    n = 400
    regressors = rng.standard_normal((n, 1, 1))
    truth = LGSSM(
        F=np.ones((1, 1)), Q=[[0.01]], H=regressors, R=[[0.2]],
        m0=[1.0], P0=[[0.5]],
    )
    _, obs = simulate(truth, n, rng)

    start = LGSSM(
        F=np.ones((1, 1)), Q=[[1.0]], H=regressors, R=[[1.0]],
        m0=[1.0], P0=[[0.5]],
    )
    result = fit_em(start, obs, estimate=("Q", "R"), max_iter=100)
    assert np.all(np.diff(result.loglik) >= -1e-8)
    assert result.model.R[0, 0] == pytest.approx(0.2, rel=0.3)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_time_varying_parameters_cannot_be_estimated():
    rng = np.random.default_rng(7)
    model = LGSSM(
        F=np.ones((1, 1)), Q=[[0.1]], H=rng.standard_normal((20, 1, 1)),
        R=[[0.3]], m0=[0.0], P0=[[1.0]],
    )
    _, obs = simulate(model, 20, rng)
    with pytest.raises(ValueError, match="H is time-varying"):
        fit_em(model, obs, estimate=("H",))


def test_unknown_parameter_name_is_rejected(rng):
    truth = scalar_model()
    _, obs = simulate(truth, 10, rng)
    with pytest.raises(ValueError, match="cannot estimate"):
        fit_em(truth, obs, estimate=("Q", "sigma"))


def test_missing_data_blocks_the_measurement_updates(rng):
    truth = scalar_model()
    _, obs = simulate(truth, 50, rng)
    obs[7, 0] = np.nan

    with pytest.raises(NotImplementedError, match="missing"):
        fit_em(truth, obs, estimate=("R",))

    # ... but the state-side parameters are fine with gaps.
    result = fit_em(truth, obs, estimate=("Q",), max_iter=10)
    assert np.all(np.diff(result.loglik) >= -1e-8)


def test_max_iter_is_respected(rng):
    truth = scalar_model()
    _, obs = simulate(truth, 200, rng)
    result = fit_em(scalar_model(Q=4.0, R=4.0), obs, max_iter=3, tol=0.0)
    assert result.n_iter == 3
    assert not result.converged
    assert result.loglik.shape == (4,)

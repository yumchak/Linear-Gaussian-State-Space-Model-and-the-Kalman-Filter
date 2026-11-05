"""Tests for the RTS smoother.

Same strategy as the filter's tests: the smoothing distribution is the
dense posterior given *all* the data, so :mod:`lgssm.batch` is the oracle
again.  The lag-one cross-covariances are checked against the off-diagonal
blocks of that same dense posterior, which is the only part of the
smoother's output the recursion does not share with the marginals.
"""

import numpy as np
import pytest

from lgssm import LGSSM, batch_posterior, kalman_filter, rts_smoother, simulate

FORMS = ["standard", "joseph", "sqrt"]


def smooth(model, y, form="joseph"):
    return rts_smoother(model, kalman_filter(model, y, form=form))


# ---------------------------------------------------------------------------
# exactness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", FORMS)
def test_smoother_matches_batch(model, data, form):
    """p(x_t | y_{1:T}) from the backward pass equals the dense posterior."""
    result = smooth(model, data, form)
    batch = batch_posterior(model, data)
    np.testing.assert_allclose(result.means, batch.means, atol=1e-10)
    np.testing.assert_allclose(result.covs, batch.covs, atol=1e-10)


@pytest.mark.parametrize("form", FORMS)
def test_smoothed_initial_state_matches_batch(model, data, form):
    """The extra backward step onto the prior gives p(x_0 | y_{1:T})."""
    result = smooth(model, data, form)
    batch = batch_posterior(model, data, include_initial=True)
    np.testing.assert_allclose(result.initial_mean, batch.means[0], atol=1e-10)
    np.testing.assert_allclose(result.initial_cov, batch.covs[0], atol=1e-10)


@pytest.mark.parametrize("form", FORMS)
def test_cross_covariances_match_batch(model, data, form):
    """Cov(x_i, x_{i+1} | y_{1:T}), against the dense joint's own blocks."""
    result = smooth(model, data, form)
    batch = batch_posterior(model, data, include_initial=True)
    d = model.dim_x
    for i in range(len(data)):
        block = batch.joint_cov[i * d : (i + 1) * d, (i + 1) * d : (i + 2) * d]
        np.testing.assert_allclose(result.cross_covs[i], block, atol=1e-10)


def test_forms_agree(model, data):
    results = {form: smooth(model, data, form) for form in FORMS}
    ref = results["joseph"]
    for form, other in results.items():
        np.testing.assert_allclose(other.means, ref.means, atol=1e-10, err_msg=form)
        np.testing.assert_allclose(other.covs, ref.covs, atol=1e-10, err_msg=form)
        np.testing.assert_allclose(
            other.cross_covs, ref.cross_covs, atol=1e-10, err_msg=form
        )


def test_scalar_random_walk_closed_form():
    """Two steps of a scalar random walk, against the algebra done by hand.

    With P0 = 0 the state starts known, so after one measurement the filter
    and the smoother can both be written out in closed form.
    """
    q, r = 1.0, 1.0
    model = LGSSM(
        F=np.ones((1, 1)), Q=[[q]], H=np.ones((1, 1)), R=[[r]], m0=[0.0], P0=[[0.0]]
    )
    y = np.array([[2.0], [4.0]])
    result = smooth(model, y)

    # Filtering at t=1: P'_1 = q, S = q + r, so m_1 = y_1/2 and P_1 = 1/2.
    # Smoothing at t=1 then folds in y_2 through G_1 = P_1 / P'_2 = (1/2)/(3/2).
    p1, p_pred2 = q * r / (q + r), q * r / (q + r) + q
    m1, m_pred2 = y[0, 0] * q / (q + r), y[0, 0] * q / (q + r)
    gain = p1 / p_pred2
    s2 = p_pred2 + r
    m2 = m_pred2 + (p_pred2 / s2) * (y[1, 0] - m_pred2)

    assert result.means[1, 0] == pytest.approx(m2)
    assert result.means[0, 0] == pytest.approx(m1 + gain * (m2 - m_pred2))
    assert result.initial_mean[0] == pytest.approx(0.0)
    assert result.initial_cov[0, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# properties the output must have
# ---------------------------------------------------------------------------


def test_smoother_ends_where_the_filter_ends(model, data):
    """At the final time there is no future data, so the two must coincide."""
    filtered = kalman_filter(model, data)
    smoothed = rts_smoother(model, filtered)
    np.testing.assert_allclose(smoothed.means[-1], filtered.means[-1], atol=1e-12)
    np.testing.assert_allclose(smoothed.covs[-1], filtered.covs[-1], atol=1e-12)


def test_smoothing_cannot_increase_uncertainty(model, data):
    """Using more data cannot widen the posterior, at any t."""
    filtered = kalman_filter(model, data)
    smoothed = rts_smoother(model, filtered)
    for t in range(len(data)):
        diff = filtered.covs[t] - smoothed.covs[t]
        assert np.linalg.eigvalsh(diff).min() > -1e-10
    # ... and strictly, wherever there is future data to use.
    assert np.trace(smoothed.covs[0]) < np.trace(filtered.covs[0])


@pytest.mark.parametrize("form", FORMS)
def test_covariances_are_symmetric_and_psd(model, data, form):
    result = smooth(model, data, form)
    for t, cov in enumerate(result.covs):
        np.testing.assert_allclose(cov, cov.T, atol=1e-12, err_msg=f"t={t}")
        assert np.linalg.eigvalsh(cov).min() > -1e-10, f"t={t}"


def test_smoother_beats_filter_on_average(rng, make_model):
    """The point of the smoother, as a statement about RMSE."""
    model = make_model(rng)
    states, obs = simulate(model, 300, rng)
    filtered = kalman_filter(model, obs)
    smoothed = rts_smoother(model, filtered)

    truth = states[1:]
    filter_rmse = np.sqrt(np.mean((filtered.means - truth) ** 2))
    smoother_rmse = np.sqrt(np.mean((smoothed.means - truth) ** 2))
    assert smoother_rmse < filter_rmse


# ---------------------------------------------------------------------------
# model variations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", FORMS)
def test_missing_data(model, data, form):
    """Gaps are handled on the way forward; the backward pass sees moments."""
    y = data.copy()
    y[2, 0] = np.nan
    y[5, :] = np.nan

    result = smooth(model, y, form)
    batch = batch_posterior(model, y)
    np.testing.assert_allclose(result.means, batch.means, atol=1e-10)
    np.testing.assert_allclose(result.covs, batch.covs, atol=1e-10)


def test_all_missing_returns_the_prior_pushed_forward(model, data):
    """With no data at all, smoothing has nothing to add to the prior."""
    y = np.full_like(data, np.nan)
    result = smooth(model, y)
    filtered = kalman_filter(model, y)
    np.testing.assert_allclose(result.means, filtered.means, atol=1e-10)
    np.testing.assert_allclose(result.initial_mean, model.m0, atol=1e-10)
    np.testing.assert_allclose(result.initial_cov, model.P0, atol=1e-10)


@pytest.mark.parametrize("form", FORMS)
def test_time_varying_matches_constant(model, data, form):
    n = len(data)
    tiled = LGSSM(
        F=np.repeat(model.F[None], n, axis=0),
        Q=np.repeat(model.Q[None], n, axis=0),
        H=np.repeat(model.H[None], n, axis=0),
        R=np.repeat(model.R[None], n, axis=0),
        m0=model.m0,
        P0=model.P0,
    )
    ref = smooth(model, data, form)
    out = smooth(tiled, data, form)
    np.testing.assert_allclose(out.means, ref.means, atol=1e-12)
    np.testing.assert_allclose(out.covs, ref.covs, atol=1e-12)


@pytest.mark.parametrize("form", FORMS)
def test_singular_predicted_covariance(tracking_model, form):
    """A known initial state and near-deterministic position dynamics."""
    _, obs = simulate(tracking_model, 10, np.random.default_rng(7))
    result = smooth(tracking_model, obs, form)
    batch = batch_posterior(tracking_model, obs)
    np.testing.assert_allclose(result.means, batch.means, atol=1e-8)
    np.testing.assert_allclose(result.covs, batch.covs, atol=1e-8)


def test_deterministic_dynamics(form="joseph"):
    """Q = 0 makes the predicted covariance singular, so the gain is not
    unique; the least-squares fallback must still give the right answer."""
    model = LGSSM(
        F=np.array([[1.0, 1.0], [0.0, 1.0]]),
        Q=np.zeros((2, 2)),
        H=np.array([[1.0, 0.0]]),
        R=[[0.3]],
        m0=[0.0, 0.0],
        P0=np.eye(2),
    )
    _, obs = simulate(model, 6, np.random.default_rng(3))
    result = smooth(model, obs, form)
    batch = batch_posterior(model, obs)
    np.testing.assert_allclose(result.means, batch.means, atol=1e-8)
    np.testing.assert_allclose(result.covs, batch.covs, atol=1e-8)


def test_single_step(model):
    """With one measurement, smoothing is filtering."""
    _, obs = simulate(model, 1, np.random.default_rng(11))
    filtered = kalman_filter(model, obs)
    smoothed = rts_smoother(model, filtered)
    np.testing.assert_allclose(smoothed.means, filtered.means, atol=1e-12)
    assert smoothed.cross_covs.shape == (1, model.dim_x, model.dim_x)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_dimension_mismatch_is_rejected(rng, make_model, data, model):
    """A filter result from a different model must not be smoothed."""
    other = make_model(rng, d=3, k=2)
    _, other_obs = simulate(other, 8, rng)
    result = kalman_filter(other, other_obs)
    with pytest.raises(ValueError, match="dimensional"):
        rts_smoother(model, result)


def test_too_many_steps_for_time_varying_model_is_rejected(model, data):
    filtered = kalman_filter(model, data)
    tiled = LGSSM(
        F=np.repeat(model.F[None], 3, axis=0),
        Q=model.Q, H=model.H, R=model.R, m0=model.m0, P0=model.P0,
    )
    with pytest.raises(ValueError, match="defined for 3 steps"):
        rts_smoother(tiled, filtered)

"""Tests for the Kalman filter.

The main check is against :mod:`lgssm.batch`, which conditions a dense
joint Gaussian in one step.  Because the filter is exact, the two must
agree to machine precision -- not approximately, and not only at the end:
the filtering distribution at time t is the batch posterior computed from
the prefix y_{1:t}, so that comparison is made at every t.
"""

import numpy as np
import pytest

from lgssm import LGSSM, batch_posterior, forecast, kalman_filter, simulate

FORMS = ["standard", "joseph", "sqrt"]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def random_model(rng, d=2, k=2, scale=0.5):
    """A random, reasonably well-conditioned LGSSM."""
    F = np.eye(d) + scale * rng.standard_normal((d, d)) / d
    H = rng.standard_normal((k, d))
    Q = _random_cov(rng, d, scale=0.3)
    R = _random_cov(rng, k, scale=0.5)
    P0 = _random_cov(rng, d, scale=1.0)
    m0 = rng.standard_normal(d)
    return LGSSM(F=F, Q=Q, H=H, R=R, m0=m0, P0=P0)


def _random_cov(rng, n, scale=1.0):
    A = rng.standard_normal((n, n))
    return scale * (A @ A.T + n * np.eye(n))


@pytest.fixture
def rng():
    return np.random.default_rng(20261021)


@pytest.fixture
def model(rng):
    return random_model(rng)


@pytest.fixture
def data(rng, model):
    _, obs = simulate(model, 8, rng)
    return obs


# ---------------------------------------------------------------------------
# exactness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", FORMS)
def test_filter_matches_batch_at_every_step(model, data, form):
    """p(x_t | y_{1:t}) from the filter equals the dense posterior."""
    result = kalman_filter(model, data, form=form)
    for t in range(1, len(data) + 1):
        batch = batch_posterior(model, data[:t])
        np.testing.assert_allclose(result.means[t - 1], batch.means[-1], atol=1e-10)
        np.testing.assert_allclose(result.covs[t - 1], batch.covs[-1], atol=1e-10)


@pytest.mark.parametrize("form", FORMS)
def test_loglik_matches_batch(model, data, form):
    """The prediction-error decomposition gives the dense density."""
    result = kalman_filter(model, data, form=form)
    batch = batch_posterior(model, data)
    assert result.loglik == pytest.approx(batch.loglik, abs=1e-9)


@pytest.mark.parametrize("form", FORMS)
def test_predicted_moments_match_batch(model, data, form):
    """p(x_t | y_{1:t-1}) is the posterior from the shorter prefix, pushed on."""
    result = kalman_filter(model, data, form=form)
    np.testing.assert_allclose(result.pred_means[0], model.F @ model.m0, atol=1e-12)
    for t in range(2, len(data) + 1):
        batch = batch_posterior(model, data[: t - 1])
        mean, cov = forecast(model, batch.means[-1], batch.covs[-1], 1, start=t - 1)
        np.testing.assert_allclose(result.pred_means[t - 1], mean[0], atol=1e-10)
        np.testing.assert_allclose(result.pred_covs[t - 1], cov[0], atol=1e-10)


def test_forms_agree(model, data):
    """The three updates are the same algorithm written three ways."""
    results = {form: kalman_filter(model, data, form=form) for form in FORMS}
    ref = results["joseph"]
    for form, other in results.items():
        np.testing.assert_allclose(other.means, ref.means, atol=1e-10, err_msg=form)
        np.testing.assert_allclose(other.covs, ref.covs, atol=1e-10, err_msg=form)
        assert other.loglik == pytest.approx(ref.loglik, abs=1e-9), form


def test_scalar_closed_form():
    """One step of the scalar filter, against the algebra done by hand."""
    q, r, p0, y1 = 0.3, 0.7, 2.0, 1.4
    model = LGSSM(
        F=np.ones((1, 1)), Q=[[q]], H=np.ones((1, 1)), R=[[r]], m0=[0.0], P0=[[p0]]
    )
    result = kalman_filter(model, np.array([[y1]]))

    p_pred = p0 + q
    s = p_pred + r
    gain = p_pred / s
    assert result.means[0, 0] == pytest.approx(gain * y1)
    assert result.covs[0, 0, 0] == pytest.approx(p_pred * r / s)
    assert result.loglik == pytest.approx(
        -0.5 * (np.log(2 * np.pi * s) + y1**2 / s)
    )


# ---------------------------------------------------------------------------
# missing data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", FORMS)
def test_all_missing_is_pure_prediction(model, data, form):
    """A fully missing row must leave the prediction untouched."""
    y = np.full_like(data, np.nan)
    result = kalman_filter(model, y, form=form)
    means, covs = forecast(model, model.m0, model.P0, len(y))

    np.testing.assert_allclose(result.means, means, atol=1e-10)
    np.testing.assert_allclose(result.covs, covs, atol=1e-10)
    assert result.loglik == 0.0


@pytest.mark.parametrize("form", FORMS)
def test_partially_missing_matches_batch(model, data, form):
    """Dropping one component of a measurement, not the whole measurement."""
    y = data.copy()
    y[2, 0] = np.nan
    y[5, :] = np.nan

    result = kalman_filter(model, y, form=form)
    batch = batch_posterior(model, y)
    np.testing.assert_allclose(result.means[-1], batch.means[-1], atol=1e-10)
    np.testing.assert_allclose(result.covs[-1], batch.covs[-1], atol=1e-10)
    assert result.loglik == pytest.approx(batch.loglik, abs=1e-9)

    assert result.innovations[2].size == 1
    assert result.innovations[5].size == 0
    assert result.loglik_terms[5] == 0.0


def test_missing_data_widens_uncertainty(model, data):
    """Throwing a measurement away cannot sharpen the posterior."""
    y = data.copy()
    y[3, :] = np.nan
    full = kalman_filter(model, data)
    gappy = kalman_filter(model, y)
    assert np.trace(gappy.covs[3]) > np.trace(full.covs[3])


# ---------------------------------------------------------------------------
# model variations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", FORMS)
def test_time_varying_matches_constant(rng, model, data, form):
    """Tiling constant parameters over time must change nothing."""
    n = len(data)
    tiled = LGSSM(
        F=np.repeat(model.F[None], n, axis=0),
        Q=np.repeat(model.Q[None], n, axis=0),
        H=np.repeat(model.H[None], n, axis=0),
        R=np.repeat(model.R[None], n, axis=0),
        m0=model.m0,
        P0=model.P0,
    )
    assert tiled.n_steps == n

    ref = kalman_filter(model, data, form=form)
    out = kalman_filter(tiled, data, form=form)
    np.testing.assert_allclose(out.means, ref.means, atol=1e-12)
    np.testing.assert_allclose(out.covs, ref.covs, atol=1e-12)


@pytest.mark.parametrize("form", FORMS)
def test_drift_matches_batch(rng, model, data, form):
    """A known drift only shifts the prediction mean."""
    drifted = LGSSM(
        F=model.F, Q=model.Q, H=model.H, R=model.R, m0=model.m0, P0=model.P0,
        u=np.array([0.4, -0.2]),
    )
    result = kalman_filter(drifted, data, form=form)
    batch = batch_posterior(drifted, data)
    np.testing.assert_allclose(result.means[-1], batch.means[-1], atol=1e-10)
    np.testing.assert_allclose(result.covs[-1], batch.covs[-1], atol=1e-10)

    undrifted = kalman_filter(model, data, form=form)
    assert not np.allclose(result.means, undrifted.means)
    np.testing.assert_allclose(result.covs, undrifted.covs, atol=1e-12)


@pytest.mark.parametrize("form", FORMS)
def test_singular_prior_covariance(form):
    """A known initial state gives P0 = 0, which must not break anything."""
    model = LGSSM(
        F=np.array([[1.0, 1.0], [0.0, 1.0]]),
        Q=np.diag([1e-4, 0.1]),
        H=np.array([[1.0, 0.0]]),
        R=[[0.5]],
        m0=[0.0, 1.0],
        P0=np.zeros((2, 2)),
    )
    rng = np.random.default_rng(7)
    _, obs = simulate(model, 10, rng)
    result = kalman_filter(model, obs, form=form)

    batch = batch_posterior(model, obs)
    np.testing.assert_allclose(result.means[-1], batch.means[-1], atol=1e-9)
    np.testing.assert_allclose(result.covs[-1], batch.covs[-1], atol=1e-9)


def test_one_dimensional_y_accepted():
    """Scalar measurements may be passed as a flat array."""
    model = LGSSM(
        F=np.ones((1, 1)), Q=[[0.1]], H=np.ones((1, 1)), R=[[0.2]], m0=[0.0], P0=[[1.0]]
    )
    flat = kalman_filter(model, np.array([1.0, 2.0, 3.0]))
    column = kalman_filter(model, np.array([[1.0], [2.0], [3.0]]))
    np.testing.assert_allclose(flat.means, column.means)


# ---------------------------------------------------------------------------
# properties the output must have
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", FORMS)
def test_covariances_are_symmetric_and_psd(model, data, form):
    result = kalman_filter(model, data, form=form)
    for name, covs in (("filtered", result.covs), ("predicted", result.pred_covs)):
        for t, cov in enumerate(covs):
            np.testing.assert_allclose(cov, cov.T, atol=1e-12, err_msg=f"{name} t={t}")
            assert np.linalg.eigvalsh(cov).min() > -1e-10, f"{name} t={t}"


@pytest.mark.parametrize("form", FORMS)
def test_update_does_not_increase_uncertainty(model, data, form):
    """Conditioning on a measurement can only shrink the covariance."""
    result = kalman_filter(model, data, form=form)
    for t in range(len(data)):
        diff = result.pred_covs[t] - result.covs[t]
        assert np.linalg.eigvalsh(diff).min() > -1e-10


def test_filter_beats_the_prior_on_average(rng):
    """A sanity check on the whole pipeline, not a property of one run."""
    model = random_model(rng)
    states, obs = simulate(model, 200, rng)
    result = kalman_filter(model, obs)

    truth = states[1:]
    filtered_rmse = np.sqrt(np.mean((result.means - truth) ** 2))
    predicted_rmse = np.sqrt(np.mean((result.pred_means - truth) ** 2))
    assert filtered_rmse < predicted_rmse


def test_loglik_terms_sum_to_loglik(model, data):
    result = kalman_filter(model, data)
    assert result.loglik == pytest.approx(result.loglik_terms.sum())


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------


def test_simulate_shapes_and_reproducibility(model):
    states, obs = simulate(model, 12, np.random.default_rng(1))
    assert states.shape == (13, model.dim_x)
    assert obs.shape == (12, model.dim_y)

    again, obs_again = simulate(model, 12, np.random.default_rng(1))
    np.testing.assert_allclose(states, again)
    np.testing.assert_allclose(obs, obs_again)


def test_simulate_respects_the_dynamics():
    """With no noise the simulation is deterministic, so it can be checked."""
    model = LGSSM(
        F=np.array([[1.0, 1.0], [0.0, 1.0]]),
        Q=np.zeros((2, 2)),
        H=np.eye(2),
        R=1e-12 * np.eye(2),
        m0=[0.0, 2.0],
        P0=np.zeros((2, 2)),
        u=np.array([0.5, 0.0]),
    )
    states, _ = simulate(model, 3, np.random.default_rng(0))
    expected = np.array([[0.0, 2.0], [2.5, 2.0], [5.0, 2.0], [7.5, 2.0]])
    np.testing.assert_allclose(states, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# forecasting
# ---------------------------------------------------------------------------


def test_forecast_agrees_with_filtering_on_missing_data(model, data):
    """Forecasting h steps is filtering h missing measurements."""
    result = kalman_filter(model, data)
    means, covs = forecast(model, result.means[-1], result.covs[-1], 4, start=len(data))

    padded = np.vstack([data, np.full((4, model.dim_y), np.nan)])
    extended = kalman_filter(model, padded)
    np.testing.assert_allclose(extended.means[len(data):], means, atol=1e-10)
    np.testing.assert_allclose(extended.covs[len(data):], covs, atol=1e-10)


def test_forecast_rejects_nonpositive_horizon(model):
    with pytest.raises(ValueError, match="horizon"):
        forecast(model, model.m0, model.P0, 0)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _valid_kwargs():
    return dict(
        F=np.eye(2), Q=np.eye(2), H=np.eye(2), R=np.eye(2),
        m0=np.zeros(2), P0=np.eye(2),
    )


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("F", np.eye(3), "F must have shape"),
        ("Q", np.array([[1.0, 0.5], [0.0, 1.0]]), "Q must be symmetric"),
        ("Q", -np.eye(2), "Q must be positive semi-definite"),
        ("R", np.zeros((2, 2)), "R must be positive definite"),
        ("P0", np.ones((3, 3)), "P0 must have shape"),
        ("m0", np.zeros((2, 1)), "m0 must be 1-D"),
        ("H", np.zeros((2, 3)), "H must have shape"),
        ("u", np.zeros(3), "u must have shape"),
    ],
)
def test_invalid_model_is_rejected(field, value, match):
    kwargs = _valid_kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        LGSSM(**kwargs)


def test_inconsistent_time_axes_are_rejected():
    kwargs = _valid_kwargs()
    kwargs["F"] = np.repeat(np.eye(2)[None], 3, axis=0)
    kwargs["Q"] = np.repeat(np.eye(2)[None], 4, axis=0)
    with pytest.raises(ValueError, match="disagree on T"):
        LGSSM(**kwargs)


def test_unknown_form_is_rejected(model, data):
    with pytest.raises(ValueError, match="unknown form"):
        kalman_filter(model, data, form="ensemble")


def test_wrong_observation_width_is_rejected(model, data):
    with pytest.raises(ValueError, match="y must have shape"):
        kalman_filter(model, data[:, :1])


def test_too_many_steps_for_time_varying_model_is_rejected(model, data):
    tiled = LGSSM(
        F=np.repeat(model.F[None], 3, axis=0),
        Q=model.Q, H=model.H, R=model.R, m0=model.m0, P0=model.P0,
    )
    with pytest.raises(ValueError, match="defined for 3 steps"):
        kalman_filter(tiled, data)

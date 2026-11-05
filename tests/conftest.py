"""Fixtures shared by the filter and smoother tests."""

import numpy as np
import pytest

from lgssm import LGSSM, simulate


def _random_cov(rng, n, scale=1.0):
    A = rng.standard_normal((n, n))
    return scale * (A @ A.T + n * np.eye(n))


@pytest.fixture
def make_model():
    """A factory for random, reasonably well-conditioned LGSSMs."""

    def _make(rng, d=2, k=2, scale=0.5):
        return LGSSM(
            F=np.eye(d) + scale * rng.standard_normal((d, d)) / d,
            Q=_random_cov(rng, d, scale=0.3),
            H=rng.standard_normal((k, d)),
            R=_random_cov(rng, k, scale=0.5),
            m0=rng.standard_normal(d),
            P0=_random_cov(rng, d, scale=1.0),
        )

    return _make


@pytest.fixture
def rng():
    return np.random.default_rng(20261021)


@pytest.fixture
def model(rng, make_model):
    return make_model(rng)


@pytest.fixture
def data(rng, model):
    _, obs = simulate(model, 8, rng)
    return obs


@pytest.fixture
def tracking_model():
    """A near-deterministic model: constant velocity, position observed."""
    return LGSSM(
        F=np.array([[1.0, 1.0], [0.0, 1.0]]),
        Q=np.diag([1e-4, 0.1]),
        H=np.array([[1.0, 0.0]]),
        R=[[0.5]],
        m0=[0.0, 1.0],
        P0=np.zeros((2, 2)),
    )

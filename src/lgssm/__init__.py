"""Kalman filtering for linear-Gaussian state space models.

    from lgssm import LGSSM, simulate, kalman_filter

    model = LGSSM(F=F, Q=Q, H=H, R=R, m0=m0, P0=P0)
    states, obs = simulate(model, 100, rng=np.random.default_rng(0))
    result = kalman_filter(model, obs)

``result.means`` and ``result.covs`` are the filtering distributions, and
``result.loglik`` the log-likelihood of the data under the model.  Passing
that result to ``rts_smoother(model, result)`` gives the smoothing
distributions p(x_t | y_{1:T}).
"""

from .batch import BatchResult, batch_posterior
from .em import EMResult, fit_em
from .kalman import (
    LGSSM,
    FilterResult,
    SmootherResult,
    forecast,
    kalman_filter,
    rts_smoother,
    simulate,
)

__all__ = [
    "LGSSM",
    "FilterResult",
    "SmootherResult",
    "simulate",
    "kalman_filter",
    "rts_smoother",
    "forecast",
    "BatchResult",
    "batch_posterior",
    "EMResult",
    "fit_em",
]

__version__ = "0.1.0"

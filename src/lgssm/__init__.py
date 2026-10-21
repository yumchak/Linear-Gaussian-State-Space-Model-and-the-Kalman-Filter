"""Kalman filtering for linear-Gaussian state space models.

    from lgssm import LGSSM, simulate, kalman_filter

    model = LGSSM(F=F, Q=Q, H=H, R=R, m0=m0, P0=P0)
    states, obs = simulate(model, 100, rng=np.random.default_rng(0))
    result = kalman_filter(model, obs)

``result.means`` and ``result.covs`` are the filtering distributions, and
``result.loglik`` the log-likelihood of the data under the model.
"""

from .batch import BatchResult, batch_posterior
from .kalman import LGSSM, FilterResult, forecast, kalman_filter, simulate

__all__ = [
    "LGSSM",
    "FilterResult",
    "simulate",
    "kalman_filter",
    "forecast",
    "BatchResult",
    "batch_posterior",
]

__version__ = "0.1.0"

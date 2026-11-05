"""Was the filter right, and did it know how sure it was?

Accuracy and calibration are different questions, and a filter can pass
one while failing the other.  :func:`rmse` answers the first: how far the
estimates fall from the truth.  The rest answer the second: an estimate
that comes with a covariance is claiming to know its own error, and that
claim is checkable.

The central quantity is the normalised estimation error squared,

    eps_t = (x_t - m_t)' P_t^{-1} (x_t - m_t),

which under a correctly specified model is chi-squared on d degrees of
freedom, whatever the model is (Bar-Shalom, Li and Kirubarajan, 2001).
So E[eps_t] = d, and a filter reporting eps much larger than d is
overconfident -- its covariances are too small for the errors it is
actually making -- while eps much smaller means it is needlessly timid.
That makes NEES the natural test of a misspecified Q or R, where the
estimates may still look reasonable but the error bars do not.

On real data there is no true state to compare against, and NEES cannot
be computed at all.  :func:`nis` is the measurement-space substitute: the
innovations are checkable because the measurements are observed.

Everything here is numpy-only, including the chi-squared quantiles, which
are computed from the regularised incomplete gamma function rather than
taken from a table.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .kalman import LGSSM, FilterResult, _solve_psd, kalman_filter, rts_smoother, simulate

__all__ = [
    "rmse",
    "nees",
    "nis",
    "coverage",
    "joint_coverage",
    "consistency_interval",
    "monte_carlo_nees",
    "chi2_cdf",
    "chi2_quantile",
]


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------


def rmse(estimates: np.ndarray, truth: np.ndarray, per_dimension: bool = False):
    """Root mean squared error of ``estimates`` against ``truth``.

    Parameters
    ----------
    estimates : (T, d) array
        Filtered or smoothed means.
    truth : (T, d) array
        The true states.  :func:`lgssm.simulate` returns T + 1 of them,
        including the draw from the prior, so pass ``states[1:]``.
    per_dimension : bool
        Return one value per state component instead of one overall.

    Returns
    -------
    float, or (d,) array if ``per_dimension``
    """
    estimates, truth = _aligned(estimates, truth)
    squared = (estimates - truth) ** 2
    if per_dimension:
        return np.sqrt(squared.mean(axis=0))
    return float(np.sqrt(squared.mean()))


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def nees(means: np.ndarray, covs: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Normalised estimation error squared, one value per time step.

    Under a correctly specified model each value is a draw from
    chi-squared on d degrees of freedom, so the sequence should average
    around d.  Works on filtered or smoothed output alike.
    """
    means, truth = _aligned(means, truth)
    covs = np.asarray(covs, dtype=float)
    if covs.shape[0] != means.shape[0]:
        raise ValueError(
            f"got {covs.shape[0]} covariances for {means.shape[0]} means"
        )

    errors = truth - means
    return np.array(
        [float(e @ _solve_psd(P, e)) for e, P in zip(errors, covs)]
    )


def nis(result: FilterResult) -> Tuple[np.ndarray, np.ndarray]:
    """Normalised innovation squared, and its degrees of freedom.

    The innovation v_t = y_t - H m'_t has covariance S_t under the model,
    so v_t' S_t^{-1} v_t is chi-squared on k degrees of freedom -- and,
    unlike NEES, it needs no true state.  This is the consistency check
    available on real data.

    Under a correctly specified model the innovation sequence is white,
    so unlike NEES these samples are independent across time and the test
    in :func:`consistency_interval` is exact within a single run.

    Steps with missing measurements contribute fewer degrees of freedom,
    and a fully missing step contributes none, hence the second return
    value: sum it rather than assuming ``T * k``.

    Returns
    -------
    values : (T,) array
    dofs : (T,) integer array
        How many measurement components entered each value.
    """
    values = np.empty(len(result.innovations))
    dofs = np.empty(len(result.innovations), dtype=int)
    for t, (v, S) in enumerate(zip(result.innovations, result.innovation_covs)):
        dofs[t] = v.size
        values[t] = 0.0 if v.size == 0 else float(v @ _solve_psd(S, v))
    return values, dofs


def coverage(
    means: np.ndarray, covs: np.ndarray, truth: np.ndarray, level: float = 0.95
) -> np.ndarray:
    """Fraction of the time each component's credible interval contains the truth.

    The interval is the marginal one, m +/- z sqrt(P_ii), so a
    well-calibrated filter returns something close to ``level`` in every
    component.  Marginal intervals ignore the correlation between
    components; :func:`joint_coverage` does not.
    """
    means, truth = _aligned(means, truth)
    covs = np.asarray(covs, dtype=float)
    z = math.sqrt(chi2_quantile(level, 1))

    sigma = np.sqrt(np.diagonal(covs, axis1=1, axis2=2))
    inside = np.abs(truth - means) <= z * sigma
    return inside.mean(axis=0)


def joint_coverage(
    means: np.ndarray, covs: np.ndarray, truth: np.ndarray, level: float = 0.95
) -> float:
    """Fraction of the time the truth falls in the joint credible ellipsoid.

    The ellipsoid is {x : (x - m)' P^{-1} (x - m) <= q}, with q the
    ``level`` quantile of chi-squared on d degrees of freedom -- so this is
    a statement about the whole covariance, not just its diagonal.
    """
    values = nees(means, covs, truth)
    return float(np.mean(values <= chi2_quantile(level, means.shape[1])))


def consistency_interval(
    dof: int, n_samples: int, level: float = 0.95
) -> Tuple[float, float]:
    """Acceptance interval for the average of ``n_samples`` chi-squared draws.

    The usual NEES or NIS test: if the average falls outside this interval,
    the model's own uncertainty is not consistent with the errors observed.

    The test assumes the samples are independent, which is true across
    independent Monte Carlo runs (see :func:`monte_carlo_nees`) and only
    approximately true across time within a single run, where neighbouring
    states are correlated.  Averaging down a column of
    :func:`monte_carlo_nees` keeps it exact.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    total = dof * n_samples
    tail = (1.0 - level) / 2.0
    return (
        chi2_quantile(tail, total) / n_samples,
        chi2_quantile(1.0 - tail, total) / n_samples,
    )


def monte_carlo_nees(
    truth_model: LGSSM,
    n_steps: int,
    n_runs: int,
    filter_model: Optional[LGSSM] = None,
    rng: Optional[np.random.Generator] = None,
    smooth: bool = False,
) -> np.ndarray:
    """NEES over independent replications, for a calibration study.

    Each run simulates a fresh trajectory from ``truth_model`` and filters
    it with ``filter_model``, which defaults to the same thing.  Passing a
    different one is the misspecification experiment: simulate under the
    truth, filter under the wrong Q or R, and watch NEES leave its
    acceptance interval.

    Returns
    -------
    (n_runs, n_steps) array
        Column t holds n_runs independent draws of the NEES at time t.
    """
    if rng is None:
        rng = np.random.default_rng()
    if filter_model is None:
        filter_model = truth_model

    out = np.empty((n_runs, n_steps))
    for run in range(n_runs):
        states, obs = simulate(truth_model, n_steps, rng)
        result = kalman_filter(filter_model, obs)
        if smooth:
            smoothed = rts_smoother(filter_model, result)
            out[run] = nees(smoothed.means, smoothed.covs, states[1:])
        else:
            out[run] = nees(result.means, result.covs, states[1:])
    return out


# ---------------------------------------------------------------------------
# the chi-squared distribution, without scipy
# ---------------------------------------------------------------------------


def chi2_cdf(x: float, dof: int) -> float:
    """P(X <= x) for X chi-squared on ``dof`` degrees of freedom."""
    if dof < 1:
        raise ValueError("dof must be at least 1")
    if x <= 0.0:
        return 0.0
    return _gamma_p(dof / 2.0, x / 2.0)


def chi2_quantile(p: float, dof: int) -> float:
    """The ``p`` quantile of chi-squared on ``dof`` degrees of freedom.

    Found by bisection on :func:`chi2_cdf`, which is accurate to roughly
    machine precision and quite fast enough for the handful of quantiles a
    diagnostic needs.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between 0 and 1")
    if dof < 1:
        raise ValueError("dof must be at least 1")

    low, high = 0.0, max(float(dof), 1.0)
    while chi2_cdf(high, dof) < p:
        high *= 2.0
        if high > 1e12:  # pragma: no cover - p is bounded away from 1
            raise RuntimeError("failed to bracket the quantile")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if chi2_cdf(middle, dof) < p:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _gamma_p(a: float, x: float) -> float:
    """The regularised lower incomplete gamma function P(a, x).

    Series expansion below the transition point and a continued fraction
    above it, which is the standard pairing: each converges quickly in the
    region where the other does not.
    """
    if x < a + 1.0:
        return _gamma_series(a, x)
    return 1.0 - _gamma_continued_fraction(a, x)


def _gamma_series(a: float, x: float) -> float:
    term = 1.0 / a
    total = term
    n = a
    for _ in range(1000):
        n += 1.0
        term *= x / n
        total += term
        if abs(term) < abs(total) * 1e-16:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_continued_fraction(a: float, x: float) -> float:
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1001):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _aligned(estimates: np.ndarray, truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    estimates = np.atleast_2d(np.asarray(estimates, dtype=float))
    truth = np.atleast_2d(np.asarray(truth, dtype=float))
    if truth.shape == (estimates.shape[0] + 1, estimates.shape[1]):
        raise ValueError(
            "truth has one row more than the estimates: simulate() returns the "
            "prior draw as states[0], so pass states[1:]"
        )
    if estimates.shape != truth.shape:
        raise ValueError(
            f"estimates have shape {estimates.shape}, truth {truth.shape}"
        )
    return estimates, truth

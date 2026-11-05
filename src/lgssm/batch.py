"""A brute-force posterior for linear-Gaussian state space models.

This exists only so the tests have something to check the Kalman filter
against.  It assembles the joint covariance of (x_{1:T}, y_{1:T}) as one
dense matrix and conditions on the measurements in a single step, which is
the definition of the answer rather than a recursion that computes it.  The
cost is O((Td)^3), so it is for small T only -- that gap is exactly what the
filter buys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .kalman import LGSSM

__all__ = ["BatchResult", "batch_posterior"]


@dataclass(frozen=True)
class BatchResult:
    """Dense posterior of x_{1:T} given the observed part of y_{1:T}.

    Attributes
    ----------
    means, covs : (T, d) and (T, d, d) arrays
        Marginal posteriors of each x_t.  With all T measurements these are
        the smoothing distributions; run on a prefix y_{1:t}, the last entry
        is the filtering distribution at time t.  With ``include_initial``
        there are T + 1 entries and index 0 is x_0.
    joint_mean, joint_cov : (T*d,) and (T*d, T*d) arrays
        The full joint posterior, states stacked in time order.  Its
        off-diagonal blocks are the posterior cross-covariances, which is
        what the smoother's lag-one cross-covariances are checked against.
    loglik : float
        log p(y_{1:T}), as a plain multivariate normal density.
    """

    means: np.ndarray
    covs: np.ndarray
    joint_mean: np.ndarray
    joint_cov: np.ndarray
    loglik: float


def batch_posterior(model: LGSSM, y, include_initial: bool = False) -> BatchResult:
    """Posterior of x_{1:T} given ``y``, by dense Gaussian conditioning.

    Missing components (``nan``) are dropped from the measurement vector,
    exactly as the filter drops them from its update.

    Parameters
    ----------
    model : LGSSM
    y : (T, k) or (T,) array
    include_initial : bool
        Also return x_0, as index 0 of every array.  The smoothed initial
        state and the lag-one cross-covariances involve x_0, so testing
        them needs it; the filter's own tests do not.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    n, d = y.shape[0], model.dim_x
    offset = 1 if include_initial else 0

    prior_mean, prior_cov = _state_prior(model, n, include_initial)

    # Stack the observed measurement components into one long vector:
    # y_obs = G x + noise, with G block diagonal over time.
    rows, obs_vals, noise_blocks = [], [], []
    for i in range(n):
        idx = np.flatnonzero(~np.isnan(y[i]))
        if idx.size == 0:
            continue
        G_i = np.zeros((idx.size, (n + offset) * d))
        block = i + offset
        G_i[:, block * d : (block + 1) * d] = model.H_at(i)[idx]
        rows.append(G_i)
        obs_vals.append(y[i, idx])
        noise_blocks.append(model.R_at(i)[np.ix_(idx, idx)])

    if not rows:
        joint_mean, joint_cov = prior_mean, prior_cov
        loglik = 0.0
    else:
        G = np.vstack(rows)
        y_obs = np.concatenate(obs_vals)
        R_all = _block_diag(noise_blocks)

        cov_xy = prior_cov @ G.T
        cov_yy = G @ prior_cov @ G.T + R_all
        resid = y_obs - G @ prior_mean

        gain = np.linalg.solve(cov_yy, cov_xy.T).T
        joint_mean = prior_mean + gain @ resid
        joint_cov = prior_cov - gain @ cov_xy.T
        joint_cov = 0.5 * (joint_cov + joint_cov.T)
        loglik = _gaussian_logpdf(resid, cov_yy)

    n_blocks = n + offset
    means = joint_mean.reshape(n_blocks, d)
    covs = np.stack(
        [joint_cov[i * d : (i + 1) * d, i * d : (i + 1) * d] for i in range(n_blocks)]
    )
    return BatchResult(
        means=means, covs=covs, joint_mean=joint_mean, joint_cov=joint_cov, loglik=loglik
    )


def _state_prior(
    model: LGSSM, n: int, include_initial: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of the states before seeing any measurement.

    Built over x_0, ..., x_n and then trimmed, so that the recursion for the
    cross-covariance blocks does not need a separate first case.
    """
    d = model.dim_x
    size = n + 1
    mean = np.empty((size, d))
    cov = np.zeros((size * d, size * d))

    mean[0] = model.m0
    cov[:d, :d] = model.P0
    for i in range(1, size):
        F = model.F_at(i - 1)
        mean[i] = F @ mean[i - 1] + model.u_at(i - 1)
        prev = cov[(i - 1) * d : i * d, (i - 1) * d : i * d]
        block = F @ prev @ F.T + model.Q_at(i - 1)
        cov[i * d : (i + 1) * d, i * d : (i + 1) * d] = 0.5 * (block + block.T)
        # Cov(x_j, x_i) = Cov(x_j, x_{i-1}) F' for j < i.
        for j in range(i):
            cross = cov[j * d : (j + 1) * d, (i - 1) * d : i * d] @ F.T
            cov[j * d : (j + 1) * d, i * d : (i + 1) * d] = cross
            cov[i * d : (i + 1) * d, j * d : (j + 1) * d] = cross.T

    if not include_initial:
        mean, cov = mean[1:], cov[d:, d:]
    return mean.reshape(-1), 0.5 * (cov + cov.T)


def _block_diag(blocks) -> np.ndarray:
    sizes = [b.shape[0] for b in blocks]
    out = np.zeros((sum(sizes), sum(sizes)))
    start = 0
    for b, size in zip(blocks, sizes):
        out[start : start + size, start : start + size] = b
        start += size
    return out


def _gaussian_logpdf(v: np.ndarray, S: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        raise np.linalg.LinAlgError("covariance is not positive definite")
    return float(-0.5 * (v.size * np.log(2.0 * np.pi) + logdet + v @ np.linalg.solve(S, v)))

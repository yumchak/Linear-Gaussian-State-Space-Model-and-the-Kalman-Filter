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
        is the filtering distribution at time t.
    joint_mean, joint_cov : (T*d,) and (T*d, T*d) arrays
        The full joint posterior, states stacked in time order.
    loglik : float
        log p(y_{1:T}), as a plain multivariate normal density.
    """

    means: np.ndarray
    covs: np.ndarray
    joint_mean: np.ndarray
    joint_cov: np.ndarray
    loglik: float


def batch_posterior(model: LGSSM, y) -> BatchResult:
    """Posterior of x_{1:T} given ``y``, by dense Gaussian conditioning.

    Missing components (``nan``) are dropped from the measurement vector,
    exactly as the filter drops them from its update.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    n, d = y.shape[0], model.dim_x

    prior_mean, prior_cov = _state_prior(model, n)

    # Stack the observed measurement components into one long vector:
    # y_obs = G x + noise, with G block diagonal over time.
    rows, obs_vals, noise_blocks = [], [], []
    for i in range(n):
        idx = np.flatnonzero(~np.isnan(y[i]))
        if idx.size == 0:
            continue
        G_i = np.zeros((idx.size, n * d))
        G_i[:, i * d : (i + 1) * d] = model.H_at(i)[idx]
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

    means = joint_mean.reshape(n, d)
    covs = np.stack([joint_cov[i * d : (i + 1) * d, i * d : (i + 1) * d] for i in range(n)])
    return BatchResult(
        means=means, covs=covs, joint_mean=joint_mean, joint_cov=joint_cov, loglik=loglik
    )


def _state_prior(model: LGSSM, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of (x_1, ..., x_n) before seeing any measurement."""
    d = model.dim_x
    mean = np.empty((n, d))
    cov = np.zeros((n * d, n * d))

    m, P = model.m0, model.P0
    blocks = []  # blocks[i] is Cov(x_{i+1}, x_{i+1})
    for i in range(n):
        F = model.F_at(i)
        m = F @ m + model.u_at(i)
        P = F @ P @ F.T + model.Q_at(i)
        P = 0.5 * (P + P.T)
        mean[i] = m
        blocks.append(P)
        cov[i * d : (i + 1) * d, i * d : (i + 1) * d] = P
        # Cov(x_j, x_i) = Cov(x_j, x_{i-1}) F' for j < i.
        for j in range(i):
            prev = cov[j * d : (j + 1) * d, (i - 1) * d : i * d]
            block = prev @ F.T
            cov[j * d : (j + 1) * d, i * d : (i + 1) * d] = block
            cov[i * d : (i + 1) * d, j * d : (j + 1) * d] = block.T

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

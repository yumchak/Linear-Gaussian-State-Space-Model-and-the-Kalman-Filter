"""Maximum likelihood estimation of LGSSM parameters, by EM.

Everything here assumes the setting of the board: we have a joint model
p_theta(x, y), we only observe y, and fitting theta would be easy if we
could see x as well.  Jensen's inequality turns the intractable
log p_theta(y) into a lower bound

    F(theta, q) = int q(x) log( p_theta(x, y) / q(x) ) dx,

which is tight when q is the posterior.  EM is alternating maximisation of
F: the E-step sets q to the posterior at the current parameters, and the
M-step maximises the expected complete-data log-likelihood over theta.  F
increases at every half-step by construction, which is why the observed
log-likelihood can never go down -- a property worth asserting in a test
rather than taking on trust, and :func:`fit_em` records the sequence so
that you can.

For an LGSSM the E-step is exactly a Kalman filter followed by an RTS
smoother, and the M-step is weighted least squares in closed form.  The
complete-data log-likelihood contains the cross term x_t x_{t-1}', so the
M-step also needs the lag-one posterior cross-covariances; those come from
the smoother as ``cross_covs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from .kalman import LGSSM, _as_obs_array, _symmetrise, kalman_filter, rts_smoother

__all__ = ["EMResult", "fit_em"]

ESTIMABLE = ("F", "Q", "H", "R", "m0", "P0")


@dataclass(frozen=True)
class EMResult:
    """Output of :func:`fit_em`.

    Attributes
    ----------
    model : LGSSM
        The fitted model.  Parameters that were not estimated are carried
        over from the starting model unchanged.
    loglik : (n_iter + 1,) array
        The observed-data log-likelihood at the start of each iteration,
        with the fitted model's own value last, so that ``loglik[-1]``
        always belongs to ``model``.  EM guarantees this is non-decreasing.
    n_iter : int
        Number of M-steps taken.
    converged : bool
        Whether the log-likelihood increase fell below ``tol`` before
        ``max_iter`` was reached.
    """

    model: LGSSM
    loglik: np.ndarray
    n_iter: int
    converged: bool

    @property
    def loglik_increase(self) -> float:
        """Total improvement from the starting parameters."""
        return float(self.loglik[-1] - self.loglik[0])


def fit_em(
    model: LGSSM,
    y,
    estimate: Sequence[str] = ("Q", "R"),
    max_iter: int = 100,
    tol: float = 1e-6,
) -> EMResult:
    """Fit LGSSM parameters to ``y`` by the EM algorithm.

    Parameters
    ----------
    model : LGSSM
        Starting parameters.  Anything not named in ``estimate`` is held
        fixed at its value here, so this is both the initial guess and the
        statement of what is assumed known.
    y : (T, k) or (T,) array
        Measurements, possibly with ``nan`` for missing components.
    estimate : sequence of str
        Which of ``"F"``, ``"Q"``, ``"H"``, ``"R"``, ``"m0"``, ``"P0"`` to
        estimate.  The default, ``("Q", "R")``, is the noise covariances:
        the usual unknowns, and the ones whose misspecification shows up
        most clearly in the filter's behaviour.

        Estimating ``F`` and ``H`` together is possible but rarely wise:
        the pair is only identified up to a similarity transform, since
        replacing (F, H, x) by (AFA^-1, HA^-1, Ax) gives the same
        distribution over y.  Fix one of them, or accept that which
        solution you land on depends on where you start.
    max_iter : int
        Maximum number of M-steps.
    tol : float
        Stop once the log-likelihood increases by less than this.

    Returns
    -------
    EMResult

    Raises
    ------
    ValueError
        If ``estimate`` names something unknown, or a parameter that is
        time-varying in ``model``.  A time-varying parameter has a value
        per time step and nothing to pool over, so it cannot be estimated
        from one sequence -- but it may perfectly well be *known*, as the
        measurement matrix is in a time-varying regression, and those are
        fine to hold fixed while estimating the rest.
    NotImplementedError
        If ``y`` has missing components and ``H`` or ``R`` is being
        estimated; see Notes.

    Notes
    -----
    Missing measurements are handled throughout the E-step, so ``F``,
    ``Q``, ``m0`` and ``P0`` can be estimated from gappy data.  The ``H``
    and ``R`` updates are a different matter: done properly they require
    imputing the missing components of y, which is not implemented here.

    The drift ``u`` is always treated as known.
    """
    y = _as_obs_array(model, y)
    estimate = tuple(estimate)
    _check_estimate(model, y, estimate)

    history = []
    current = model
    converged = False
    n_iter = 0

    for _ in range(max_iter):
        filtered = kalman_filter(current, y)
        history.append(filtered.loglik)
        if len(history) >= 2 and history[-1] - history[-2] <= tol:
            converged = True
            break
        smoothed = rts_smoother(current, filtered)
        current = _m_step(current, y, smoothed, estimate)
        n_iter += 1

    if not converged:
        # The last M-step has not been scored yet; score it, so that the
        # final entry always belongs to the model being returned.
        history.append(kalman_filter(current, y).loglik)

    return EMResult(
        model=current,
        loglik=np.asarray(history),
        n_iter=n_iter,
        converged=converged,
    )


def _check_estimate(model: LGSSM, y: np.ndarray, estimate: Tuple[str, ...]) -> None:
    unknown = [name for name in estimate if name not in ESTIMABLE]
    if unknown:
        raise ValueError(
            f"cannot estimate {unknown}; choose from {list(ESTIMABLE)}"
        )

    const_ndim = {"F": 2, "Q": 2, "H": 2, "R": 2}
    for name in estimate:
        if name in const_ndim and getattr(model, name).ndim != const_ndim[name]:
            raise ValueError(
                f"{name} is time-varying in this model, so it cannot be estimated; "
                "hold it fixed, or estimate a time-invariant parameter instead"
            )

    if np.isnan(y).any() and ({"H", "R"} & set(estimate)):
        raise NotImplementedError(
            "estimating H or R from data with missing components is not supported; "
            "the update would have to impute the missing measurements"
        )


def _m_step(
    model: LGSSM, y: np.ndarray, smoothed, estimate: Tuple[str, ...]
) -> LGSSM:
    """Maximise the expected complete-data log-likelihood, in closed form.

    Every update is a weighted least squares fit in which the unobserved
    states are replaced by their posterior moments: the means where the
    states appear linearly, and the second moments -- means outer means,
    *plus* the posterior covariance -- where they appear quadratically.
    That extra covariance term is the whole difference between EM and
    naively fitting to the smoothed means.
    """
    n, d = smoothed.means.shape

    # Index 0 is x_0 throughout, so that moment[t] is the moment of x_t.
    means = np.vstack([smoothed.initial_mean[None, :], smoothed.means])
    covs = np.concatenate([smoothed.initial_cov[None, :, :], smoothed.covs])

    # E[x_t x_t'], E[x_t x_{t-1}'] and E[x_{t-1} x_{t-1}'], for t = 1..n.
    second = np.empty((n, d, d))
    cross = np.empty((n, d, d))
    previous = np.empty((n, d, d))
    for t in range(1, n + 1):
        i = t - 1
        second[i] = covs[t] + np.outer(means[t], means[t])
        cross[i] = smoothed.cross_covs[i].T + np.outer(means[t], means[t - 1])
        previous[i] = covs[t - 1] + np.outer(means[t - 1], means[t - 1])

    params: Dict[str, np.ndarray] = {}

    if "F" in estimate:
        rhs = cross.sum(axis=0)
        if model.u is not None:
            rhs = rhs - np.outer(model.u_at(0), means[:-1].sum(axis=0))
        params["F"] = np.linalg.solve(previous.sum(axis=0), rhs.T).T

    def F_at(i: int) -> np.ndarray:
        return params["F"] if "F" in params else model.F_at(i)

    if "Q" in estimate:
        total = np.zeros((d, d))
        for t in range(1, n + 1):
            i, F, u = t - 1, F_at(t - 1), model.u_at(t - 1)
            # Second moment of the process noise x_t - F x_{t-1} - u.
            residual = (
                second[i] - cross[i] @ F.T - F @ cross[i].T + F @ previous[i] @ F.T
            )
            drift = means[t] - F @ means[t - 1]
            residual += np.outer(u, u) - np.outer(u, drift) - np.outer(drift, u)
            total += residual
        params["Q"] = _symmetrise(total / n)

    if "H" in estimate:
        rhs = sum(np.outer(y[i], means[i + 1]) for i in range(n))
        params["H"] = np.linalg.solve(second.sum(axis=0), rhs.T).T

    def H_at(i: int) -> np.ndarray:
        return params["H"] if "H" in params else model.H_at(i)

    if "R" in estimate:
        k = model.dim_y
        total = np.zeros((k, k))
        for i in range(n):
            H = H_at(i)
            innovation = y[i] - H @ means[i + 1]
            total += np.outer(innovation, innovation) + H @ covs[i + 1] @ H.T
        params["R"] = _symmetrise(total / n)

    if "m0" in estimate:
        params["m0"] = smoothed.initial_mean
    if "P0" in estimate:
        # The mean term vanishes when m0 is estimated too, and does not
        # when m0 is held fixed away from the smoothed initial state.
        offset = smoothed.initial_mean - params.get("m0", model.m0)
        params["P0"] = _symmetrise(smoothed.initial_cov + np.outer(offset, offset))

    return LGSSM(
        F=params.get("F", model.F),
        Q=params.get("Q", model.Q),
        H=params.get("H", model.H),
        R=params.get("R", model.R),
        m0=params.get("m0", model.m0),
        P0=params.get("P0", model.P0),
        u=model.u,
    )

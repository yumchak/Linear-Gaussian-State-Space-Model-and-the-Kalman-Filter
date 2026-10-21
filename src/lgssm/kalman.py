"""The Kalman filter for linear-Gaussian state space models.

The model is

    x_t = F_t x_{t-1} + u_t + q_t,   q_t ~ N(0, Q_t),
    y_t = H_t x_t + r_t,             r_t ~ N(0, R_t),
    x_0 ~ N(m0, P0),

for t = 1, ..., T, with the noises and the prior mutually independent.

Indexing convention
-------------------
Observations are held in an array ``y`` of shape ``(T, k)``, where ``y[i]``
is the measurement y_{i+1}; the prior lives at t = 0 and has no measurement.
Time-varying parameters follow the same 0-based step index: ``F[i]`` takes
x_i to x_{i+1}, and ``H[i]``, ``R[i]`` apply to y_{i+1}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

__all__ = ["LGSSM", "FilterResult", "simulate", "kalman_filter", "forecast"]


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LGSSM:
    """A linear-Gaussian state space model.

    Each of ``F``, ``Q``, ``H``, ``R`` and ``u`` is either a single array
    (constant in time) or an array with a leading time axis of length T
    (time-varying).  Mixing the two is allowed, as long as every time axis
    present has the same length.

    Parameters
    ----------
    F : (d, d) or (T, d, d) array
        Transition matrix.
    Q : (d, d) or (T, d, d) array
        Process noise covariance; symmetric positive semi-definite.
    H : (k, d) or (T, k, d) array
        Measurement matrix.
    R : (k, k) or (T, k, k) array
        Measurement noise covariance; symmetric positive definite.
    m0 : (d,) array
        Prior mean of x_0.
    P0 : (d, d) array
        Prior covariance of x_0; symmetric positive semi-definite.
    u : (d,) or (T, d) array, optional
        Known additive drift in the dynamics.  Defaults to zero.
    """

    F: np.ndarray
    Q: np.ndarray
    H: np.ndarray
    R: np.ndarray
    m0: np.ndarray
    P0: np.ndarray
    u: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        for name in ("F", "Q", "H", "R", "m0", "P0"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float))
        if self.u is not None:
            object.__setattr__(self, "u", np.asarray(self.u, dtype=float))
        self._validate()

    # -- shape helpers ------------------------------------------------------

    @property
    def dim_x(self) -> int:
        """Dimension d of the hidden state."""
        return self.m0.shape[0]

    @property
    def dim_y(self) -> int:
        """Dimension k of a measurement."""
        return self.H.shape[-2]

    @property
    def n_steps(self) -> Optional[int]:
        """Length T of the time axis, or None if every parameter is constant."""
        for arr, const_ndim in ((self.F, 2), (self.Q, 2), (self.H, 2), (self.R, 2)):
            if arr.ndim == const_ndim + 1:
                return arr.shape[0]
        if self.u is not None and self.u.ndim == 2:
            return self.u.shape[0]
        return None

    def _at(self, arr: Optional[np.ndarray], const_ndim: int, i: int) -> Optional[np.ndarray]:
        if arr is None:
            return None
        return arr if arr.ndim == const_ndim else arr[i]

    def F_at(self, i: int) -> np.ndarray:
        """Transition matrix taking x_i to x_{i+1}."""
        return self._at(self.F, 2, i)

    def Q_at(self, i: int) -> np.ndarray:
        """Process noise covariance of the step from x_i to x_{i+1}."""
        return self._at(self.Q, 2, i)

    def H_at(self, i: int) -> np.ndarray:
        """Measurement matrix of y_{i+1}."""
        return self._at(self.H, 2, i)

    def R_at(self, i: int) -> np.ndarray:
        """Measurement noise covariance of y_{i+1}."""
        return self._at(self.R, 2, i)

    def u_at(self, i: int) -> np.ndarray:
        """Known drift of the step from x_i to x_{i+1}."""
        if self.u is None:
            return np.zeros(self.dim_x)
        return self._at(self.u, 1, i)

    # -- validation ---------------------------------------------------------

    def _validate(self) -> None:
        if self.m0.ndim != 1:
            raise ValueError(f"m0 must be 1-D, got shape {self.m0.shape}")
        d = self.dim_x
        if self.P0.shape != (d, d):
            raise ValueError(f"P0 must have shape {(d, d)}, got {self.P0.shape}")

        _check_param_shape("F", self.F, (d, d))
        _check_param_shape("Q", self.Q, (d, d))
        if self.H.ndim not in (2, 3):
            raise ValueError(f"H must be 2-D or 3-D, got shape {self.H.shape}")
        k = self.dim_y
        _check_param_shape("H", self.H, (k, d))
        _check_param_shape("R", self.R, (k, k))
        if self.u is not None:
            _check_param_shape("u", self.u, (d,))

        varying = [(self.F, 2), (self.Q, 2), (self.H, 2), (self.R, 2)]
        if self.u is not None:
            varying.append((self.u, 1))
        lengths = {arr.shape[0] for arr, const_ndim in varying if arr.ndim == const_ndim + 1}
        if len(lengths) > 1:
            raise ValueError(f"time-varying parameters disagree on T: {sorted(lengths)}")

        _check_cov("P0", self.P0)
        _check_cov("Q", self.Q)
        _check_cov("R", self.R, strict=True)


def _check_param_shape(name: str, arr: np.ndarray, const_shape: Tuple[int, ...]) -> None:
    const_ndim = len(const_shape)
    if arr.ndim == const_ndim:
        if arr.shape != const_shape:
            raise ValueError(f"{name} must have shape {const_shape}, got {arr.shape}")
    elif arr.ndim == const_ndim + 1:
        if arr.shape[1:] != const_shape:
            raise ValueError(
                f"time-varying {name} must have shape (T, *{const_shape}), got {arr.shape}"
            )
    else:
        raise ValueError(
            f"{name} must have shape {const_shape} or (T, *{const_shape}), got {arr.shape}"
        )


def _check_cov(name: str, arr: np.ndarray, strict: bool = False) -> None:
    """Check that ``arr`` is (a stack of) symmetric PSD matrices.

    Definiteness is judged by whether a Cholesky factorisation succeeds,
    rather than against a fixed eigenvalue threshold, so that a genuinely
    tiny measurement noise is not mistaken for a singular one.
    """
    mats = arr if arr.ndim == 3 else arr[None]
    for i, mat in enumerate(mats):
        where = f"{name}[{i}]" if arr.ndim == 3 else name
        scale = max(1.0, float(np.abs(mat).max()))
        if not np.allclose(mat, mat.T, atol=1e-10 * scale):
            raise ValueError(f"{where} must be symmetric")
        if strict:
            try:
                np.linalg.cholesky(mat)
            except np.linalg.LinAlgError:
                raise ValueError(f"{where} must be positive definite") from None
        else:
            eigvals = np.linalg.eigvalsh(mat)
            tol = 1e-8 * max(1.0, float(np.abs(eigvals).max()))
            if eigvals.min() < -tol:
                raise ValueError(f"{where} must be positive semi-definite")


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------


def simulate(
    model: LGSSM, n_steps: int, rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw a trajectory and its measurements from ``model``.

    Returns
    -------
    states : (n_steps + 1, d) array
        The true states x_0, ..., x_T.  ``states[0]`` is the draw from the
        prior and has no measurement, so ``states[1:]`` lines up with ``obs``
        and with the arrays returned by :func:`kalman_filter`.
    obs : (n_steps, k) array
        The measurements y_1, ..., y_T.
    """
    if rng is None:
        rng = np.random.default_rng()
    _check_n_steps(model, n_steps)

    d, k = model.dim_x, model.dim_y
    states = np.empty((n_steps + 1, d))
    obs = np.empty((n_steps, k))
    states[0] = rng.multivariate_normal(model.m0, model.P0, method="svd")
    for i in range(n_steps):
        q = rng.multivariate_normal(np.zeros(d), model.Q_at(i), method="svd")
        states[i + 1] = model.F_at(i) @ states[i] + model.u_at(i) + q
        r = rng.multivariate_normal(np.zeros(k), model.R_at(i), method="svd")
        obs[i] = model.H_at(i) @ states[i + 1] + r
    return states, obs


# ---------------------------------------------------------------------------
# filtering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterResult:
    """Output of a forward pass of the Kalman filter.

    All arrays are indexed by the same 0-based step index as ``y``: entry
    ``i`` refers to time t = i + 1.

    Attributes
    ----------
    means, covs : (T, d) and (T, d, d) arrays
        The filtering distributions p(x_t | y_{1:t}).
    pred_means, pred_covs : (T, d) and (T, d, d) arrays
        The one-step prediction distributions p(x_t | y_{1:t-1}).  Kept
        because the RTS smoother needs them.
    innovations, innovation_covs : lists of arrays
        The innovation y_t - H_t m'_t and its covariance S_t.  Entries are
        shorter than k at steps where some measurements were missing, and
        empty where all were.
    loglik_terms : (T,) array
        Per-step log p(y_t | y_{1:t-1}); zero where y_t was fully missing.
    loglik : float
        The log-likelihood log p(y_{1:T}), by the prediction-error
        decomposition.  This is the hook for parameter estimation.
    form : str
        Which update was used: "standard", "joseph" or "sqrt".
    """

    means: np.ndarray
    covs: np.ndarray
    pred_means: np.ndarray
    pred_covs: np.ndarray
    innovations: List[np.ndarray]
    innovation_covs: List[np.ndarray]
    loglik_terms: np.ndarray
    loglik: float
    form: str


def kalman_filter(model: LGSSM, y, form: str = "joseph") -> FilterResult:
    """Run the Kalman filter over the measurements ``y``.

    Parameters
    ----------
    model : LGSSM
    y : (T, k) or (T,) array
        Measurements y_1, ..., y_T.  A ``nan`` marks a missing component:
        that component is dropped from the update, so a fully ``nan`` row
        leaves a pure prediction step.
    form : {"joseph", "standard", "sqrt"}
        How the covariance update is computed.  All three are exact in
        exact arithmetic and agree to within rounding; they differ in how
        well they preserve symmetry and positive definiteness.

        - ``"standard"``: P = (I - K H) P', the textbook form, cheapest.
        - ``"joseph"``: P = (I - K H) P' (I - K H)' + K R K', symmetric and
          PSD by construction.  The default.
        - ``"sqrt"``: propagates a square root of P through QR
          factorisations, so P is PSD to machine precision; the most robust
          and the slowest.

    Returns
    -------
    FilterResult
    """
    if form not in ("standard", "joseph", "sqrt"):
        raise ValueError(f"unknown form {form!r}")
    y = _as_obs_array(model, y)
    n = y.shape[0]
    _check_n_steps(model, n)

    if form == "sqrt":
        return _filter_sqrt(model, y)
    return _filter_moment(model, y, form)


def _filter_moment(model: LGSSM, y: np.ndarray, form: str) -> FilterResult:
    d = model.dim_x
    n = y.shape[0]
    means = np.empty((n, d))
    covs = np.empty((n, d, d))
    pred_means = np.empty((n, d))
    pred_covs = np.empty((n, d, d))
    innovations: List[np.ndarray] = []
    innovation_covs: List[np.ndarray] = []
    loglik_terms = np.zeros(n)

    m, P = model.m0, model.P0
    for i in range(n):
        F, Q = model.F_at(i), model.Q_at(i)
        m_pred = F @ m + model.u_at(i)
        P_pred = _symmetrise(F @ P @ F.T + Q)
        pred_means[i], pred_covs[i] = m_pred, P_pred

        idx = np.flatnonzero(~np.isnan(y[i]))
        if idx.size == 0:
            m, P = m_pred, P_pred
            innovations.append(np.zeros(0))
            innovation_covs.append(np.zeros((0, 0)))
        else:
            H = model.H_at(i)[idx]
            R = model.R_at(i)[np.ix_(idx, idx)]
            v = y[i, idx] - H @ m_pred
            S = _symmetrise(H @ P_pred @ H.T + R)
            K = np.linalg.solve(S, H @ P_pred).T

            m = m_pred + K @ v
            if form == "standard":
                P = _symmetrise((np.eye(d) - K @ H) @ P_pred)
            else:
                A = np.eye(d) - K @ H
                P = _symmetrise(A @ P_pred @ A.T + K @ R @ K.T)

            innovations.append(v)
            innovation_covs.append(S)
            loglik_terms[i] = _gaussian_logpdf(v, S)

        means[i], covs[i] = m, P

    return FilterResult(
        means=means,
        covs=covs,
        pred_means=pred_means,
        pred_covs=pred_covs,
        innovations=innovations,
        innovation_covs=innovation_covs,
        loglik_terms=loglik_terms,
        loglik=float(loglik_terms.sum()),
        form=form,
    )


def _filter_sqrt(model: LGSSM, y: np.ndarray) -> FilterResult:
    """Array square-root filter.

    Carries a matrix C with P = C C' instead of P itself.  Each step is a QR
    factorisation of a block array whose triangular factor holds the
    innovation Cholesky factor, the gain and the updated square root at
    once, so P never has a chance to lose positive definiteness.
    """
    d = model.dim_x
    n = y.shape[0]
    means = np.empty((n, d))
    covs = np.empty((n, d, d))
    pred_means = np.empty((n, d))
    pred_covs = np.empty((n, d, d))
    innovations: List[np.ndarray] = []
    innovation_covs: List[np.ndarray] = []
    loglik_terms = np.zeros(n)

    m = model.m0
    C = _psd_sqrt(model.P0)
    for i in range(n):
        F = model.F_at(i)
        m_pred = F @ m + model.u_at(i)
        # [F C, sqrt(Q)] has outer product F P F' + Q.
        pre = np.hstack([F @ C, _psd_sqrt(model.Q_at(i))])
        C_pred = _rq_lower(pre)
        P_pred = _symmetrise(C_pred @ C_pred.T)
        pred_means[i], pred_covs[i] = m_pred, P_pred

        idx = np.flatnonzero(~np.isnan(y[i]))
        if idx.size == 0:
            m, C = m_pred, C_pred
            innovations.append(np.zeros(0))
            innovation_covs.append(np.zeros((0, 0)))
        else:
            H = model.H_at(i)[idx]
            R = model.R_at(i)[np.ix_(idx, idx)]
            v = y[i, idx] - H @ m_pred
            k = idx.size

            # The triangular factor of this block array is
            #     [[chol(S),  0 ],
            #      [K chol(S), C]].
            block = np.block([
                [_psd_sqrt(R), H @ C_pred],
                [np.zeros((d, k)), C_pred],
            ])
            tri = _rq_lower(block)
            L_S = tri[:k, :k]
            KL_S = tri[k:, :k]
            C = tri[k:, k:]

            K = np.linalg.solve(L_S.T, KL_S.T).T
            m = m_pred + K @ v
            z = np.linalg.solve(L_S, v)

            S = _symmetrise(L_S @ L_S.T)
            innovations.append(v)
            innovation_covs.append(S)
            loglik_terms[i] = -0.5 * (
                k * np.log(2.0 * np.pi) + 2.0 * np.log(np.diag(L_S)).sum() + z @ z
            )

        means[i], covs[i] = m, _symmetrise(C @ C.T)

    return FilterResult(
        means=means,
        covs=covs,
        pred_means=pred_means,
        pred_covs=pred_covs,
        innovations=innovations,
        innovation_covs=innovation_covs,
        loglik_terms=loglik_terms,
        loglik=float(loglik_terms.sum()),
        form="sqrt",
    )


def forecast(
    model: LGSSM, mean: np.ndarray, cov: np.ndarray, horizon: int, start: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Push a state distribution ``horizon`` steps into the future.

    With ``mean``, ``cov`` the filtering distribution at time t, this gives
    p(x_{t+h} | y_{1:t}) for h = 1, ..., ``horizon``: the prediction task,
    run without any measurements to condition on.

    Parameters
    ----------
    model : LGSSM
    mean, cov : (d,) and (d, d) arrays
        The distribution to propagate.
    horizon : int
        Number of steps to look ahead.
    start : int
        Number of transitions already applied, so that time-varying
        parameters are picked up from the right index.  After filtering T
        measurements this is T.  Ignored for time-invariant models.

    Returns
    -------
    means : (horizon, d) array
    covs : (horizon, d, d) array
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    _check_n_steps(model, start + horizon)

    means = np.empty((horizon, model.dim_x))
    covs = np.empty((horizon, model.dim_x, model.dim_x))
    m, P = mean, cov
    for h in range(horizon):
        i = start + h
        F = model.F_at(i)
        m = F @ m + model.u_at(i)
        P = _symmetrise(F @ P @ F.T + model.Q_at(i))
        means[h], covs[h] = m, P
    return means, covs


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _as_obs_array(model: LGSSM, y) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        if model.dim_y != 1:
            raise ValueError(
                f"y must have shape (T, {model.dim_y}), got {y.shape}"
            )
        y = y[:, None]
    if y.ndim != 2 or y.shape[1] != model.dim_y:
        raise ValueError(f"y must have shape (T, {model.dim_y}), got {y.shape}")
    return y


def _check_n_steps(model: LGSSM, n: int) -> None:
    T = model.n_steps
    if T is not None and n > T:
        raise ValueError(
            f"model has time-varying parameters defined for {T} steps, asked for {n}"
        )


def _symmetrise(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def _psd_sqrt(A: np.ndarray) -> np.ndarray:
    """A matrix C with C C' = A, for symmetric PSD A.

    Uses the Cholesky factor when A is positive definite, and falls back to
    a symmetric eigendecomposition otherwise, so that singular covariances
    (a known initial state, say, or deterministic components of the
    dynamics) are handled without a jitter hack.
    """
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.clip(eigvals, 0.0, None)
        return eigvecs * np.sqrt(eigvals)


def _rq_lower(A: np.ndarray) -> np.ndarray:
    """Lower-triangular C with C C' = A A', via QR of A'.

    The sign convention of the QR factorisation is fixed so that C has a
    non-negative diagonal, which makes the Cholesky factors it produces
    unique.
    """
    r = np.linalg.qr(A.T, mode="reduced")[1]
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return (signs[:, None] * r).T


def _gaussian_logpdf(v: np.ndarray, S: np.ndarray) -> float:
    """log N(v; 0, S), with S symmetric positive definite."""
    L = _psd_sqrt(S)
    z = np.linalg.solve(L, v)
    return float(
        -0.5 * (v.size * np.log(2.0 * np.pi) + 2.0 * np.log(np.diag(L)).sum() + z @ z)
    )

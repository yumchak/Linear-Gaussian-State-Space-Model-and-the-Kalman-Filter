# Linear-Gaussian State Space Models and the Kalman Filter

Final-year group project, University of Bristol.
Victor Sze, Aseem Kulkarni, Nigel Prem Lim, Lozko Pilev — supervised by Dr Sam Power.

Started 1 October 2026, at the first supervision meeting; this repository was created on
14 October 2026.

A hidden state observed only through noisy measurements can never be known exactly, so Bayesian
filtering tracks a distribution for it instead, updated as each measurement arrives. The
linear-Gaussian case is the one setting where this can be solved exactly — by the Kalman filter —
which makes it the benchmark everything else here is tested against.

## Model

The hidden state $x_t \in \mathbb{R}^d$ and measurement $y_t \in \mathbb{R}^k$ satisfy

$$x_t = F x_{t-1} + q_t, \quad q_t \sim \mathcal{N}(0, Q),$$
$$y_t = H x_t + r_t, \quad r_t \sim \mathcal{N}(0, R),$$
$$x_0 \sim \mathcal{N}(m_0, P_0),$$

with $F, Q, H, R$ known (parameter estimation is one of the extensions). A known constant drift $u$
can be added to the dynamics, which only shifts the prediction mean.

The three inference tasks are **filtering** $p(x_t \mid y_{1:t})$, **prediction**
$p(x_{t+h} \mid y_{1:t})$ and **smoothing** $p(x_t \mid y_{1:T})$.

## Scope

Derive, prove and implement the two exact algorithms, then test them in a simulation study:

- the **Kalman filter** — forward recursion for filtering and prediction;
- the **RTS smoother** — backward pass that folds in future data.

Alongside the derivations: exactness and batch equivalence, minimum-mean-squared-error optimality,
stability, likelihood evaluation via the prediction-error decomposition, and the $O(d^3 + k^3)$
per-step cost against $O((Td)^3)$ for the batch solution.

## Code

A shared, tested Python package, reused by every extension. Install it editable, so the
extensions can `import lgssm` from wherever they live:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

```python
import numpy as np
from lgssm import LGSSM, simulate, kalman_filter, rts_smoother, fit_em

model = LGSSM(F=F, Q=Q, H=H, R=R, m0=m0, P0=P0)
states, obs = simulate(model, 100, rng=np.random.default_rng(0))

result = kalman_filter(model, obs)
result.means          # filtering distributions p(x_t | y_1:t)
result.covs
result.loglik         # log p(y_1:T), the hook for parameter estimation

smoothed = rts_smoother(model, result)
smoothed.means        # smoothing distributions p(x_t | y_1:T)
smoothed.covs
smoothed.cross_covs   # lag-one correlations, for the EM M-step

fit = fit_em(model, obs, estimate=("Q", "R"))
fit.model             # parameters fitted by maximum likelihood
fit.loglik            # one entry per iteration, guaranteed non-decreasing
```

`LGSSM` takes any of $F, Q, H, R$ as either a single matrix or a stack with a leading time axis,
so time-varying models need no separate code path. A `nan` in `y` marks a missing component and
is dropped from the update, so a fully missing measurement leaves a pure prediction step.
`kalman_filter` takes `form="standard" | "joseph" | "sqrt"` — the textbook update, the
symmetric-by-construction Joseph form, and a QR-based square-root filter that keeps the
covariance positive definite to machine precision.

**In the package now:** `kalman.py` (model, validation, simulation, filter, smoother,
forecasting), `em.py` (parameter estimation), `diagnostics.py` (accuracy and calibration) and
`batch.py`, a deliberately slow $O((Td)^3)$ dense posterior that exists only as a test oracle.
**Still to come:** the worked examples.

### Parameter estimation

Dropping the assumption that $F, Q, H, R$ are known, `fit_em` maximises the likelihood by
alternating maximisation of the free energy
$F(\theta, q) = \int q(x) \log \frac{p_\theta(x, y)}{q(x)}\,dx$: the E-step sets $q$ to the
posterior — a Kalman filter followed by an RTS smoother — and the M-step maximises the expected
complete-data log-likelihood in closed form. Because the complete-data log-likelihood contains
the cross term $x_t x_{t-1}^\top$, the M-step needs the smoother's lag-one correlations
$P_{t,t+1|T}$ as well as $m_{t|T}$ and $P_{t|T}$.

`estimate` says which parameters to fit, defaulting to the noise covariances $(Q, R)$. A
time-varying parameter cannot be estimated from one sequence, but may be *known*, as $H_t$ is in
a drifting regression — so those models fit fine. $F$ and $H$ are only jointly identified up to
a similarity transform; the docstring says so, and you should fix one of them.

EM guarantees the likelihood never decreases, which the tests assert on every fit rather than
taking on trust. The M-step updates are checked against ordinary least squares on models rigged
so the states are effectively known.

Tests double as numerical evidence for the theory:

1. **Exactness** — agreement with the batch posterior to machine precision, at every $t$, for the filter and the smoother alike. *Done.*
2. **Accuracy** — RMSE of filter and smoother against the true simulated state, via `rmse`. *Done.*
3. **Calibration** — credible interval coverage and NEES, via `coverage`, `joint_coverage`, `nees` and `nis`. *Done.*
4. **Scalability and robustness** — run time as $d$ and $k$ grow, missing observations, time-varying models. *Missing data and time-varying models done; timings still to come.*

### Diagnostics

Accuracy and calibration are separate questions: a filter can track the state well and still be
wrong about how sure it is. `rmse` answers the first. For the second, the normalised estimation
error squared

$$\epsilon_t = (x_t - m_t)^\top P_t^{-1} (x_t - m_t) \sim \chi^2_d$$

should average $d$ under a correct model, so a filter reporting much more than $d$ is
overconfident. `consistency_interval` gives the acceptance interval for the test, and
`monte_carlo_nees` runs the independent replications that make it exact — including the
misspecification experiment, where you simulate under one model and filter under another.

On real data there is no true state and NEES cannot be computed. `nis` is the substitute:
innovations are observable, and under a correct model they are white, so the same test applies
within a single run.

The chi-squared quantiles are computed from the regularised incomplete gamma function rather
than read from a table, and are tested against published values.

Running examples: a two-dimensional drifting object, vehicle tracking, and a drifting regression
coefficient such as a hedge ratio.

## Individual extensions

| Author | Extension |
| --- | --- |
| Aseem Kulkarni | **Pairs trading** — model the ratio between two related stocks as an LGSSM, backtest against simple benchmarks, possibly many pairs at once. |
| Victor Sze | **Interacting particle systems** — $N$ agents coupled through their mean, $x^{(i)}_t = A x^{(i)}_{t-1} + B \bar{x}_{t-1} + q^{(i)}_t$; a change of coordinates decouples the $Nd$-dimensional filter into independent $d$-dimensional ones, linear rather than cubic in $N$. |
| Nigel Prem Lim | **Footballers' market values** — true value as a hidden state, estimated from noisy irregular signals (on-pitch performance, published valuations, transfer fees). |
| Lozko Pilev | **Time-Varying Market Beta** — investigate how a company's exposure to the wider stock market changes over time; EM algorithm; filtering under misspecification and its link to filter stability. |

## References

- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45.
- Rauch, H. E., Tung, F. and Striebel, C. T. (1965). Maximum likelihood estimates of linear dynamic systems. *AIAA Journal*, 3(8), 1445–1450.
- Särkkä, S. and Svensson, L. (2023). *Bayesian Filtering and Smoothing*. 2nd edn. Cambridge University Press.
- Sanz-Alonso, D., Stuart, A. M. and Taeb, A. (2023). *Inverse Problems and Data Assimilation*. Cambridge University Press.
- Bar-Shalom, Y., Li, X. R. and Kirubarajan, T. (2001). *Estimation with Applications to Tracking and Navigation*. Wiley.

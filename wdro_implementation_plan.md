# WDRO Implementation Plan

## 1. Scope

Implement only the following two WDRO dual objectives from
`../../Notes/note_wdro_solve.typ`:

1. The KL/entropic dual.
2. The joint \(L^q\) dual with \(q=\infty\) and \(p=1\), giving a joint
   CVaR-like objective.

Do not implement the conditional \(L^q\) dual. In particular, the joint dual
has one global scalar threshold \(\eta\). Keeping one \(\eta_i\) per empirical
sample would instead implement the excluded conditional dual.

The pathwise function \(f_\theta(\xi')\) is the terminal distributed hedging
loss, including transaction costs and the option liability, under strategy
parameters \(\theta\).

## 2. Objectives

### Entropic WDRO

For \(a=\epsilon+\lambda\tau\), minimize jointly over the strategy and
\(\lambda\geq0\):

\[
J_{\mathrm{ent}}(\theta,\lambda)
=\lambda\rho
+a\,\mathbb E_{\xi\sim P}
\left[
\log\mathbb E_{\xi'\sim\pi_0(\cdot\mid\xi)}
\exp\left(
\frac{f_\theta(\xi')-\lambda c(\xi,\xi')}{a}
\right)
\right].
\]

Estimate each inner expectation with plain Monte Carlo and
`jax.scipy.special.logsumexp`. No correction for the finite-sample bias of the
logarithm is required. The logarithm must be taken separately for every base
sample before averaging over the base batch.

### Joint CVaR WDRO

Use \(q=\infty\) and \(p=1\). Minimize jointly over the strategy,
\(\lambda\geq0\), and one scalar \(\eta\):

\[
J_{\mathrm{joint}}(\theta,\lambda,\eta)
=\lambda\rho+\eta
+\frac{1}{1-\alpha}
\mathbb E_{\xi\sim P,\,\xi'\sim\pi_0(\cdot\mid\xi)}
\left[
\left(f_\theta(\xi')-\lambda c(\xi,\xi')-\eta\right)_+
\right].
\]

The expectation is joint across base trajectories and their perturbations.
There is no per-base-sample inner minimization and no vector of thresholds.

## 3. Trajectory representation

Represent market trajectories in unconstrained Euclidean coordinates so that
the reference distribution is normalized, sampling is simple, and proposed
paths remain valid after decoding.

- Black--Scholes coordinates: log stock values after the fixed initial value.
- Heston coordinates: log stock values and unconstrained transforms of the
  strictly positive variance values after their fixed initial values.

Decode perturbed coordinates into all quantities consumed by the hedge:

- Positive stock prices.
- Positive Heston variance.
- Heston integrated variance and variance-swap prices.
- Observable strategy features.
- Terminal call payoff.

Use location and scale statistics from a fixed standardization sample drawn
independently of training and validation to standardize the two coordinate
types before measuring transport cost. Save these statistics with the robust
state and do not recompute them per minibatch.

## 4. Cost and reference kernel

Use the quadratic transport cost

\[
c(\xi,\xi')
=\frac{1}{2}\left\|
\frac{\xi'-\xi}{\sigma_0}
\right\|_2^2.
\]

The compatible normalized reference kernel is

\[
\pi_0(\xi'\mid\xi)=\mathcal N(\xi,\sigma_0^2I),
\]

which has density proportional to \(\exp(-c(\xi,\cdot))\). The proposal scale
must be configured in standardized coordinates.

## 5. Proposal distribution

Implement \(k\)-step ULA initialized from the reference kernel. The test suite
covers exactly \(k\in\{0,1,2\}\), while the implementation accepts any
nonnegative static step count.

For \(k=0\), sample directly from

\[
\nu(\cdot\mid\xi)=\pi_0(\cdot\mid\xi),
\]

so the log importance weight is exactly zero.

For \(k>0\), use the Gaussian ULA transition

\[
K_h(y\mid x,\xi)
=\mathcal N\left(
y;
x+\frac{h}{2}\nabla_x\log\widetilde\nu(x\mid\xi),
hI
\right),
\]

where the unnormalized target is based on
\(f_\theta(x)-\lambda c(\xi,x)\). The endpoint marginal after several steps is
generally an intractable integral. Avoid evaluating it by treating the ULA
history as auxiliary proposal randomness. Conditional on the penultimate
state, use the exact log weight

\[
\log w
=\log\pi_0(x_k\mid\xi)
-\log K_h(x_k\mid x_{k-1},\xi).
\]

This augmented-space estimator is unbiased for expectations under \(\pi_0\),
although its variance can exceed that of the unavailable marginal-density
weight. Clip the ULA score norm before forming the Gaussian transition mean;
the resulting clipped-drift transition still has an explicit density. Stop
gradients through proposal samples and importance weights during
strategy optimization; they are sampling machinery for the fixed \(\pi_0\)
expectation, not trainable objective terms.

## 6. Configuration and parameter state

Add a robust configuration separate from market transaction costs:

```python
@chex.dataclass
class RobustConfig:
    """Configure the WDRO dual, reference kernel, and proposal sampling."""

    enabled: bool = False
    objective: Literal["entropic", "joint_cvar"] = "entropic"
    rho: float = 0.1
    kl_epsilon: float = 0.1
    tau: float = 0.0
    alpha: float = 0.95
    proposal_samples: int = 32
    proposal_scale: float = 0.1
    ula_steps: int = 0
    ula_step_size: float = 0.01
    ula_gradient_clip: float = 10.0
    initial_lambda: float = 1.0
    initial_eta: float = 0.0
```

Use `kl_epsilon` rather than `epsilon` to distinguish KL regularization from
the existing proportional transaction-cost parameter.

Add a small Equinox module for the dual variables:

```python
class RobustState(eqx.Module):
    """Store dual variables and fixed trajectory standardization statistics."""

    raw_lambda: Array
    eta: Array
    location: Array
    scale: Array
```

The first two leaves have scalar shape `()`. Define
`lambda_ = jax.nn.softplus(raw_lambda)` to enforce \(\lambda\geq0\). The
entropic objective ignores `eta`; the joint CVaR objective trains it.
`location` and `scale` are fixed vectors estimated once from an independent
standardization sample and receive zero optimizer gradients.

The existing `Strategy.q` is not used by either WDRO objective. Keep it only if
nominal CVaR remains available as a backward-compatible mode.

## 7. Loss and accounting refactor

Separate pathwise hedge accounting from risk aggregation.

1. Apply the strategy and compute terminal gain, transaction cost, payoff, and
   pathwise loss without reducing over samples.
2. Apply either the entropic or joint CVaR WDRO aggregator to those pathwise
   values.

For a base batch of size `batch` and `n_proposals` perturbations, use these
named shapes:

- Base coordinates: `(batch, dimension)`.
- Proposed coordinates: `(batch, n_proposals, dimension)`.
- Proposed prices:
  `(batch, n_proposals, n_time + 1, n_assets)`.
- Proposed features:
  `(batch, n_proposals, n_time, n_features)`.
- Pathwise loss: `(batch, n_proposals)`.
- Transport cost: `(batch, n_proposals)`.

Flatten the first two dimensions only while applying the strategy and hedge
accounting, then restore them before robust aggregation. Derive and name every
dimension and use `chex.assert_shape` after every tensor operation.

For the entropic objective, compute

```python
scores = (pathwise_loss - lambda_ * transport_cost) / temperature + log_weights
inner = logsumexp(scores, axis=1) - jnp.log(n_proposals)
objective = lambda_ * rho + temperature * jnp.mean(inner)
```

For joint CVaR, compute

```python
penalized = pathwise_loss - lambda_ * transport_cost
tail = jax.nn.relu(penalized - eta)
weighted_tail = jnp.exp(log_weights) * tail
objective = lambda_ * rho + eta + jnp.mean(weighted_tail) / (1.0 - alpha)
```

## 8. Cash and indifference pricing

Freeze the initial cash parameter at zero during strategy training. Both robust
risk measures are translation equivariant, so optimizing cash jointly would
make the minimization unbounded.

After training, evaluate the optimized robust objective with zero cash and use
that value as the robust indifference price. Save it in the existing `cash`
metric without feeding it back into strategy optimization.

## 9. Training and validation

- Continue drawing base training trajectories online from \(P\); a fixed
  empirical training set is unnecessary because the joint CVaR threshold is
  global.
- Draw independent proposal noise for every base sample and optimizer step.
- Use fixed validation trajectories and fixed proposal noise so validation
  values are comparable across steps.
- Optimize the strategy and `raw_lambda` jointly for both objectives.
- Optimize the scalar `eta` only for joint CVaR.
- Keep JIT only on complete training, validation, and evaluation steps.
- Use `lax.scan` for time evolution and hedge accounting.
- Treat configurations, parameters, optimizer state, coordinates, and proposal
  batches as pytrees.
- Every function must have a docstring describing its purpose, inputs, and
  outputs.

Track and save at least:

- Robust training and validation objectives.
- `lambda` and, for joint CVaR, `eta`.
- Mean transport cost.
- Pathwise hedging-loss statistics.
- Transaction cost and turnover.
- Proposal count and scale.
- Robust indifference price.

## 10. Code organization

Keep the implementation compact:

- `deep_hedging.py`: market simulation, trajectory encoding/decoding, strategy,
  accounting, robust training, evaluation, and artifacts.
- `wdro.py`: reference sampling, robust state, transport cost, and the two WDRO
  aggregators.
- `test_deep_hedging.py` and `test_wdro.py`: mathematical, shape,
  reconstruction, optimizer, artifact, and integration tests.
- `README.md`: CLI examples and definitions of the two supported robust modes.

Avoid excessive helper functions. Prefer a few complete functions with clear
named dimensions over many one-line wrappers.

## 11. Pytest coverage

Run all tests with `uv run pytest`. Add tests for:

1. Black--Scholes coordinate encode/decode consistency.
2. Heston coordinate encode/decode consistency, positive stock and variance,
   and consistent variance-swap reconstruction.
3. Reference Gaussian sampling shapes, finiteness, and fixed-key determinism.
4. Transport cost equals zero when \(\xi'=\xi\) and is nonnegative otherwise.
5. Reference-proposal importance weights are exactly one for \(k=0\), and
   augmented ULA samples and weights are finite and deterministic for
   \(k\in\{0,1,2\}\).
6. Entropic WDRO equals a hand-computed average of per-base-sample
   `log(mean(exp(...)))` values.
7. Entropic WDRO remains finite for deliberately large pathwise losses.
8. Joint CVaR WDRO equals a hand-computed objective pooled over both base and
   proposal dimensions.
9. `eta` has scalar shape, preventing accidental implementation of the
   conditional dual.
10. `lambda` is nonnegative after its parameter transformation.
11. Cash receives no training update.
12. One top-level JIT optimizer step for each objective has finite loss and
    gradients and updates the intended parameters.
13. Existing nominal Black--Scholes, Heston, accounting, artifact, and pytest
    tests continue to pass.

## 12. Implementation order

1. Add `RobustConfig` and scalar `RobustState`.
2. Refactor hedge accounting to expose unreduced pathwise losses.
3. Implement trajectory encoding and differentiable decoding for both markets.
4. Implement the standardized quadratic cost and Gaussian reference sampler.
5. Implement and unit-test the entropic aggregator.
6. Implement and unit-test the joint CVaR aggregator with one scalar `eta`.
7. Add top-level JIT training and validation steps.
8. Add robust evaluation, indifference pricing, metrics, and artifacts.
9. Add CLI documentation for both objectives.
10. Run the complete pytest suite and short deterministic smoke training for
    both markets and both robust objectives.

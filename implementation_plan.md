# Deep Hedging Implementation Plan

## 1. Scope

Implement the Deep Hedging experiment from Buehler et al. for an at-the-money European call under four configurations:

1. Black--Scholes without transaction costs.
2. Black--Scholes with proportional transaction costs.
3. Heston without transaction costs.
4. Heston with proportional transaction costs.

Use JAX, Equinox, Optax, Chex, Tyro, and `uv`. The implementation uses CVaR, the paper's simplest reported risk criterion. Robust/DRO training is outside this first implementation.

## 2. Mathematical objective

The liability is the at-the-money European call

\[
Z=(S_T^1-K)^+, \qquad K=S_0^1=100.
\]

The bank account is constant because the paper works under a risk-neutral measure with zero drift/rate. For a strategy \(\delta_k\), initial cash \(p\), and \(\delta_{-1}=0\), define

\[
G_T(\delta)=\sum_{k=0}^{n-1}\delta_k^\top(S_{k+1}-S_k),
\]

\[
C_T(\delta)=\sum_{k=0}^{n-1}\sum_{i=1}^d
\varepsilon\,S_k^i\left|\delta_k^i-\delta_{k-1}^i\right|,
\]

and

\[
e_T=p+G_T(\delta)-C_T(\delta)-Z.
\]

For pathwise loss (L_T=-e_T), train the strategy parameters and scalar VaR threshold (q) with

\[
\min_{\theta,q}\;q+\frac{1}{1-\alpha}\mathbb E[(L_T-q)^+],
\]

with default confidence level \(\alpha=0.95\). Cash is excluded from gradient updates; after training, the CVaR of the unpriced loss is used as the cash indifference price. Following the paper, no transaction cost is charged for liquidation at maturity; trading costs are charged only for changes made at \(k=0,\ldots,n-1\). The no-cost cases set \(\varepsilon=0\).

## 3. Paper settings to reproduce

### Time grid and payoff

- Horizon: \(T=30/365\).
- Trading dates: \(t_k=k/365\), \(k=0,\ldots,30\).
- Number of rebalancing decisions: \(n=30\).
- Daily rebalancing.
- Spot: \(S_0^1=100\).
- Strike: \(K=100\).

### Black--Scholes

The paper uses

\[
S_t=100\exp\left(-\tfrac12\sigma^2t+\sigma W_t\right),
\qquad \sigma=0.2.
\]

Simulate this process exactly on the daily grid. There is one hedging instrument, so \(d=1\). The observable feature is

\[
I_k=\log S_k.
\]

### Heston

Under the risk-neutral measure,

\[
dS_t^1=\sqrt{V_t}S_t^1dB_t,
\qquad
dV_t=\kappa(\bar v-V_t)dt+\xi\sqrt{V_t}dW_t,
\]

with \(d\langle B,W\rangle_t=\rho\,dt\). Use the paper's parameters:

- \(\kappa=1\).
- \(\bar v=0.04\).
- \(\rho=-0.7\).
- Volatility of variance \(\xi=2\).
- \(V_0=0.04\).
- \(S_0^1=100\).

There are two tradable hedging instruments, \(d=2\): the stock and the idealized variance swap

\[
S_t^2=\int_0^tV_sds+L(t,V_t),
\]

\[
L(t,v)=\frac{v-\bar v}{\kappa}
\left(1-e^{-\kappa(T-t)}\right)+\bar v(T-t).
\]

Use the feature vector from the paper:

\[
I_k=(\log S_k^1,V_k).
\]

Reproduce the paper's simulation description: sample each CIR transition exactly from its scaled noncentral chi-square law, approximate integrated variance as piecewise constant over a daily interval, and use the simplified Broadie--Kaya log-stock update preserving the spot/variance correlation. Accumulate the same integrated-variance approximation to construct \(S_k^2\). Add assertions that variance, stock, and variance-swap paths are finite, and document any numerical clipping needed at zero.

### Transaction costs

The paper's proportional cost is

\[
c_k(\Delta\delta_k)=\sum_i\varepsilon
|\Delta\delta_k^i|S_k^i.
\]

The main four-way comparison should use:

- No costs: \(\varepsilon=0\).
- Costs: \(\varepsilon=0.01\), matching the paper's architecture comparison.

Also expose \(\varepsilon\) in the Tyro configuration. An optional price-asymptotics sweep can reproduce the paper's grid

\[
\varepsilon_i=2^{-(i+5)},\qquad i=1,\ldots,5.
\]

## 4. Strategy parameterization

Use the paper's semi-recurrent strategy with a distinct feed-forward network at every trading date:

\[
\delta_k=F^{\theta_k}(I_k,\delta_{k-1}),
\qquad \delta_{-1}=0.
\]

For \(d\) hedging instruments, each time-specific network has:

- Input width \(N_0=2d\): the \(d\)-dimensional feature \(I_k\) concatenated with the previous \(d\)-dimensional holding.
- Two hidden layers: \(N_1=N_2=d+15\).
- Output width \(N_3=d\).
- ReLU after each hidden layer.
- Batch normalization immediately before each hidden-layer ReLU.
- Linear output layer.
- Different weights, biases, and batch-normalization state for every \(k\).

Thus Black--Scholes uses widths `2 -> 16 -> 16 -> 1`, and Heston uses `4 -> 17 -> 17 -> 2`.

Store the time-specific parameters as a pytree of arrays with the time dimension leading. Evaluate all 30 decisions with `jax.lax.scan`; do not place `jit` inside model or loss functions. The only compiled functions should be top-level training/evaluation steps.

Batch normalization must have explicit training and inference modes. Training updates running statistics; evaluation uses frozen running statistics. Since the paper does not state momentum or epsilon, configure them and default to Equinox's standard behavior. Record those defaults in run metadata.

## 5. Optimization hyperparameters

Use all hyperparameters reported by the paper:

- Optimizer: Adam.
- Learning rate: `0.005`.
- Mini-batch size: `256`.
- Random initialization of weights and biases.
- Batch normalization in every hidden layer before ReLU.
- Out-of-sample evaluation paths: `1_000_000` for the paper-scale evaluation mode.

The paper does **not** report the number of optimizer steps, training sample count, batch-normalization momentum/epsilon, random seed, or exact initializer distributions and scales. Make these explicit reproducibility choices:

- Optimizer steps: configurable, default `20_000`.
- Paths are generated online for every mini-batch, avoiding an unspecified fixed training-set size.
- Seed: configurable, default `0`.
- Equinox/JAX default linear-layer initialization, with independent keys for every date/layer.
- Batch-normalization constants: Equinox defaults, surfaced in config.
- Fast evaluation default: `100_000`; paper-scale flag: `1_000_000`, evaluated in chunks to bound memory.
- Track the best parameters using a fixed validation set and validation CVaR; default validation frequency `250` steps and validation size `65_536`.

These values are implementation defaults, not values claimed by the paper.

## 6. Code organization

Keep the implementation compact and avoid unnecessary function fragmentation:

- `pyproject.toml`: project metadata and dependencies (`jax`, `chex`, `equinox`, `optax`, `tyro`, plotting dependencies, and `pytest` as the test runner).
- `deep_hedging.py`: configs, simulators, strategy module, PnL/loss, top-level jitted train/evaluation steps, training loop, diagnostics, and Tyro entry point.
- `test_deep_hedging.py`: shape, simulation, accounting, determinism, and smoke-training tests.
- `README.md`: exact commands for all four experiments and the optional cost sweep.

Use Tyro dataclasses with nested configurations for market, strategy, costs, training, and evaluation. Select the market with a literal/enum (`black_scholes`, `heston`) and set transaction cost through `epsilon`.

Every function must have a docstring describing purpose, inputs, and outputs. Treat configurations, parameters, optimizer state, and batch-normalization state as pytrees.

## 7. JAX and shape discipline

Follow the repository JAX requirements:

- Use `chex.assert_shape` after every tensor operation.
- Derive and name dimensions such as `batch`, `n_time`, `n_assets`, and `n_features` from inputs/configuration.
- Expected market-path shapes:
  - Black--Scholes prices: `(batch, n_time + 1, 1)`.
  - Heston tradable prices: `(batch, n_time + 1, 2)`.
  - Heston variance: `(batch, n_time + 1)`.
  - Strategy holdings: `(batch, n_time, n_assets)`.
  - Payoff, gains, costs, error: `(batch,)`.
- Use `lax.scan` for time evolution in both simulation and strategy evaluation.
- Use `vmap` only for independent paths/examples where broadcasting is insufficient.
- Split random keys explicitly and never reuse a key.
- JIT only top-level train, validation, and evaluation steps.
- Use float64 when enabled for Heston transition stability; make dtype configurable and test float32 smoke execution separately.

## 8. Validation and acceptance criteria

### Unit tests

Implement and run the test suite with `pytest` (for example, `uv run pytest`).

1. Black--Scholes simulation has the required shape, positive finite prices, and deterministic output for a fixed key.
2. Heston simulation has the required shapes, nonnegative finite variance, positive finite stock, and finite variance-swap values.
3. At maturity, the variance-swap price equals accumulated integrated variance under the chosen discretization.
4. A zero strategy produces `gains == 0` and `costs == 0`.
5. A constant strategy has only its initial transaction cost.
6. Costs are zero exactly when `epsilon == 0` and nonnegative otherwise.
7. Strategy output shapes are correct for both markets, and changing previous holdings can change the output.
8. One jitted optimizer step updates parameters and returns finite loss/gradients.
9. The CVaR loss matches its defining sample formula for known losses and threshold.
10. A short training smoke test reduces validation CVaR from initialization.

### Financial sanity checks

1. Black--Scholes, no costs: fitted \(p\) should be close to the analytic Black--Scholes call price (approximately `2.29` for the paper's 30-day, 20%-volatility ATM setup), and the learned stock hedge should broadly track analytic delta.
2. Increasing the rebalance frequency in a no-cost Black--Scholes diagnostic should reduce hedging MSE.
3. Adding costs should reduce turnover relative to the no-cost learned strategy and generally increase residual hedging MSE.
4. Heston, no costs: using both stock and variance swap should outperform a diagnostic constrained to stock only.
5. Report train/validation CVaR, test CVaR and MSE diagnostics, fitted initial cash, mean/std/quantiles of hedging error, mean transaction cost, and mean turnover.

### Reproduction outputs

For each of the four configurations, save configuration, seed, final/best metrics, and model parameters. Produce:

- Training and validation loss curves.
- Test hedging-error histogram.
- Holdings/turnover summaries.
- For Black--Scholes without costs, learned versus analytic delta over spot/time grids.
- A four-case comparison table using common evaluation seeds.

## 9. Implementation order

1. Create the `uv` project and Tyro configuration tree.
2. Implement exact Black--Scholes simulation and call payoff.
3. Implement the stacked semi-recurrent network with per-date parameters and batch-normalization state.
4. Implement scanned gains, proportional costs, terminal loss, trainable VaR threshold, and CVaR objective.
5. Add the top-level jitted Adam training/evaluation steps and online batch generation.
6. Verify the Black--Scholes no-cost case against analytic price and delta.
7. Add the exact-CIR/simplified-Broadie--Kaya Heston simulator and variance-swap instrument.
8. Run and verify Heston without costs.
9. Enable `epsilon=0.01`, run both transaction-cost cases, and compare turnover/error.
10. Add chunked paper-scale evaluation, plots, saved artifacts, tests, and usage documentation.

## 10. Deliberate differences from the paper

- The loss is CVaR at configurable confidence level \(\alpha\), while MSE remains an out-of-sample diagnostic. The paper also studies entropic risk for its transaction-cost asymptotics.
- Training duration and related unspecified settings use the explicit defaults in Section 5.
- The primary cost comparison uses \(\varepsilon=0.01\); the paper's five-value asymptotic grid remains an optional experiment.
- The implementation uses JAX/Equinox/Optax rather than TensorFlow while preserving the paper's strategy architecture and reported optimizer settings.

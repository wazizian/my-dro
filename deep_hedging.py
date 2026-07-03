"""Deep-hedging experiments matching the architecture in Buehler et al."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Literal

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tyro

Array = jax.Array


@chex.dataclass
class MarketConfig:
    """Configure the simulated market; all fields are scalar model inputs."""

    name: Literal["black_scholes", "heston"] = "black_scholes"
    spot: float = 100.0
    strike: float = 100.0
    volatility: float = 0.2
    kappa: float = 1.0
    long_variance: float = 0.04
    vol_of_variance: float = 2.0
    correlation: float = -0.7
    initial_variance: float = 0.04
    n_time: int = 30
    horizon: float = 30.0 / 365.0
    dtype: Literal["float32", "float64"] = "float32"


@chex.dataclass
class StrategyConfig:
    """Configure paper-style networks; inputs are BN constants and output is architecture metadata."""

    bn_momentum: float = 0.99
    bn_epsilon: float = 1e-5
    stock_only: bool = False


@chex.dataclass
class TrainingConfig:
    """Configure optimization; fields control Adam, online batches, validation, and randomness."""

    steps: int = 20_000
    batch_size: int = 256
    learning_rate: float = 0.005
    validation_size: int = 65_536
    validation_frequency: int = 250
    seed: int = 0


@chex.dataclass
class EvaluationConfig:
    """Configure evaluation and artifacts; fields select path count, chunks, and output directory."""

    paths: int = 100_000
    chunk_size: int = 65_536
    paper_scale: bool = False
    output_dir: str = "outputs"


@chex.dataclass
class Config:
    """Collect the full experiment pytree-like configuration consumed by the CLI."""

    market: MarketConfig = dataclasses.field(default_factory=MarketConfig)
    strategy: StrategyConfig = dataclasses.field(default_factory=StrategyConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = dataclasses.field(default_factory=EvaluationConfig)
    epsilon: float = 0.0
    alpha: float = 0.95


class Strategy(eqx.Module):
    """Hold all time-specific MLP and batch-normalization parameters as leading-time arrays."""

    w1: Array
    b1: Array
    w2: Array
    b2: Array
    w3: Array
    b3: Array
    gamma1: Array
    beta1: Array
    gamma2: Array
    beta2: Array
    mean1: Array
    var1: Array
    mean2: Array
    var2: Array
    cash: Array
    q: Array


def market_dimensions(config: MarketConfig) -> tuple[int, int]:
    """Return (number of tradable assets, number of observable features) for a market config."""

    return (1, 1) if config.name == "black_scholes" else (2, 2)


def initialize_strategy(key: Array, market: MarketConfig) -> Strategy:
    """Initialize paper-width, time-specific MLPs; inputs are a PRNG key/config and output is Strategy."""

    n_assets, n_features = market_dimensions(market)
    n_time = market.n_time
    n_input = n_features + n_assets
    hidden = n_assets + 15
    dtype = jnp.float64 if market.dtype == "float64" else jnp.float32
    keys = jax.random.split(key, 6 * n_time).reshape(6, n_time, 2)
    chex.assert_shape(keys, (6, n_time, 2))
    scale1 = jnp.asarray(1.0 / np.sqrt(float(n_input)), dtype=dtype)
    chex.assert_shape(scale1, ())
    scale2 = jnp.asarray(1.0 / np.sqrt(float(hidden)), dtype=dtype)
    chex.assert_shape(scale2, ())
    w1 = jax.vmap(lambda date_key: jax.random.uniform(date_key, (hidden, n_input), minval=-scale1, maxval=scale1, dtype=dtype))(keys[0])
    chex.assert_shape(w1, (n_time, hidden, n_input))
    b1 = jax.vmap(lambda date_key: jax.random.uniform(date_key, (hidden,), minval=-scale1, maxval=scale1, dtype=dtype))(keys[1])
    chex.assert_shape(b1, (n_time, hidden))
    w2 = jax.vmap(lambda date_key: jax.random.uniform(date_key, (hidden, hidden), minval=-scale2, maxval=scale2, dtype=dtype))(keys[2])
    chex.assert_shape(w2, (n_time, hidden, hidden))
    b2 = jax.vmap(lambda date_key: jax.random.uniform(date_key, (hidden,), minval=-scale2, maxval=scale2, dtype=dtype))(keys[3])
    chex.assert_shape(b2, (n_time, hidden))
    w3 = jax.vmap(lambda date_key: jax.random.uniform(date_key, (n_assets, hidden), minval=-scale2, maxval=scale2, dtype=dtype))(keys[4])
    chex.assert_shape(w3, (n_time, n_assets, hidden))
    b3 = jax.vmap(lambda date_key: jax.random.uniform(date_key, (n_assets,), minval=-scale2, maxval=scale2, dtype=dtype))(keys[5])
    chex.assert_shape(b3, (n_time, n_assets))
    zeros1 = jnp.zeros((n_time, hidden), dtype=w1.dtype)
    chex.assert_shape(zeros1, (n_time, hidden))
    ones1 = jnp.ones((n_time, hidden), dtype=w1.dtype)
    chex.assert_shape(ones1, (n_time, hidden))
    cash = jnp.asarray(0.0, dtype=w1.dtype)
    chex.assert_shape(cash, ())
    q = jnp.asarray(0.0, dtype=w1.dtype)
    chex.assert_shape(q, ())
    return Strategy(w1, b1, w2, b2, w3, b3, ones1, zeros1, ones1, zeros1,
                    zeros1, ones1, zeros1, ones1, cash, q)


def simulate_black_scholes(key: Array, config: MarketConfig, batch: int) -> tuple[Array, Array]:
    """Simulate exact Black--Scholes paths; inputs are key/config/batch and outputs are prices/features."""

    dtype = jnp.float64 if config.dtype == "float64" else jnp.float32
    dt = jnp.asarray(config.horizon / config.n_time, dtype=dtype)
    chex.assert_shape(dt, ())
    shocks = jax.random.normal(key, (config.n_time, batch), dtype=dtype)
    chex.assert_shape(shocks, (config.n_time, batch))
    increments = (-0.5 * config.volatility**2 * dt) + config.volatility * jnp.sqrt(dt) * shocks
    chex.assert_shape(increments, (config.n_time, batch))
    log0 = jnp.full((batch,), jnp.log(config.spot), dtype=dtype)
    chex.assert_shape(log0, (batch,))
    _, log_tail = jax.lax.scan(lambda value, increment: (value + increment, value + increment), log0, increments)
    chex.assert_shape(log_tail, (config.n_time, batch))
    logs = jnp.concatenate((log0[None], log_tail), axis=0)
    chex.assert_shape(logs, (config.n_time + 1, batch))
    stock = jnp.exp(logs).T[:, :, None]
    chex.assert_shape(stock, (batch, config.n_time + 1, 1))
    features = logs.T[:, :-1, None]
    chex.assert_shape(features, (batch, config.n_time, 1))
    return stock, features


def simulate_heston(key: Array, config: MarketConfig, batch: int) -> tuple[Array, Array, Array]:
    """Simulate exact-CIR Heston variance with simplified BK stock; outputs tradables, features, variance."""

    dtype = jnp.float64 if config.dtype == "float64" else jnp.float32
    dt = jnp.asarray(config.horizon / config.n_time, dtype=dtype)
    chex.assert_shape(dt, ())
    keys = jax.random.split(key, config.n_time * 2).reshape(config.n_time, 2, 2)
    chex.assert_shape(keys, (config.n_time, 2, 2))
    kappa = jnp.asarray(config.kappa, dtype=dtype)
    chex.assert_shape(kappa, ())
    theta = jnp.asarray(config.long_variance, dtype=dtype)
    chex.assert_shape(theta, ())
    xi = jnp.asarray(config.vol_of_variance, dtype=dtype)
    chex.assert_shape(xi, ())
    rho = jnp.asarray(config.correlation, dtype=dtype)
    chex.assert_shape(rho, ())
    decay = jnp.exp(-kappa * dt)
    chex.assert_shape(decay, ())
    c = xi**2 * (1.0 - decay) / (4.0 * kappa)
    chex.assert_shape(c, ())
    dof = 4.0 * kappa * theta / xi**2
    chex.assert_shape(dof, ())

    def step(carry: tuple[Array, Array, Array], step_keys: Array):
        """Advance one Heston interval; inputs are state and two keys, output is next state and observations."""

        variance, log_stock, integrated = carry
        chex.assert_shape(variance, (batch,))
        noncentrality = decay * variance / c
        chex.assert_shape(noncentrality, (batch,))
        # A noncentral chi-square is a Poisson mixture of central chi-squares.
        # JAX has no native sampler in the pinned API, so this is still an exact
        # CIR transition rather than an Euler approximation.
        poisson_key, gamma_key = jax.random.split(step_keys[0])
        counts = jax.random.poisson(poisson_key, noncentrality / 2.0, shape=(batch,))
        chex.assert_shape(counts, (batch,))
        chi_square = 2.0 * jax.random.gamma(gamma_key, dof / 2.0 + counts, dtype=dtype)
        chex.assert_shape(chi_square, (batch,))
        next_variance = c * chi_square
        chex.assert_shape(next_variance, (batch,))
        # The paper's piecewise-constant integrated-variance approximation uses
        # the variance observed at the left endpoint of this daily interval.
        interval_variance = variance * dt
        chex.assert_shape(interval_variance, (batch,))
        z = jax.random.normal(step_keys[1], (batch,), dtype=dtype)
        chex.assert_shape(z, (batch,))
        correlated = (rho / xi) * (next_variance - variance - kappa * (theta * dt - interval_variance))
        chex.assert_shape(correlated, (batch,))
        next_log_stock = log_stock - 0.5 * interval_variance + correlated + jnp.sqrt(1.0 - rho**2) * jnp.sqrt(jnp.maximum(interval_variance, 0.0)) * z
        chex.assert_shape(next_log_stock, (batch,))
        next_integrated = integrated + interval_variance
        chex.assert_shape(next_integrated, (batch,))
        return (next_variance, next_log_stock, next_integrated), (next_variance, next_log_stock, next_integrated)

    initial = (jnp.full((batch,), config.initial_variance, dtype=dtype),
               jnp.full((batch,), jnp.log(config.spot), dtype=dtype),
               jnp.zeros((batch,), dtype=dtype))
    (_, _, _), (variance_tail, log_tail, integrated_tail) = jax.lax.scan(step, initial, keys)
    chex.assert_shape(variance_tail, (config.n_time, batch))
    variance = jnp.concatenate((initial[0][None], variance_tail), axis=0).T
    chex.assert_shape(variance, (batch, config.n_time + 1))
    logs = jnp.concatenate((initial[1][None], log_tail), axis=0).T
    chex.assert_shape(logs, (batch, config.n_time + 1))
    integrated = jnp.concatenate((initial[2][None], integrated_tail), axis=0).T
    chex.assert_shape(integrated, (batch, config.n_time + 1))
    times = jnp.linspace(0.0, config.horizon, config.n_time + 1, dtype=dtype)
    chex.assert_shape(times, (config.n_time + 1,))
    remaining = config.horizon - times[None, :]
    chex.assert_shape(remaining, (1, config.n_time + 1))
    adjustment = ((variance - theta) / kappa) * (1.0 - jnp.exp(-kappa * remaining)) + theta * remaining
    chex.assert_shape(adjustment, (batch, config.n_time + 1))
    variance_swap = integrated + adjustment
    chex.assert_shape(variance_swap, (batch, config.n_time + 1))
    stock = jnp.exp(logs)
    chex.assert_shape(stock, (batch, config.n_time + 1))
    tradables = jnp.stack((stock, variance_swap), axis=-1)
    chex.assert_shape(tradables, (batch, config.n_time + 1, 2))
    features = jnp.stack((logs[:, :-1], variance[:, :-1]), axis=-1)
    chex.assert_shape(features, (batch, config.n_time, 2))
    if not isinstance(tradables, jax.core.Tracer):
        chex.assert_tree_all_finite((tradables, variance))
        chex.assert_trees_all_equal(jnp.all(variance >= 0.0), jnp.asarray(True))
        chex.assert_trees_all_equal(jnp.all(stock > 0.0), jnp.asarray(True))
    return tradables, features, variance


def simulate_market(key: Array, config: MarketConfig, batch: int) -> tuple[Array, Array, Array | None]:
    """Dispatch market simulation; inputs are key/config/batch and outputs are tradables/features/variance."""

    if config.name == "black_scholes":
        prices, features = simulate_black_scholes(key, config, batch)
        return prices, features, None
    return simulate_heston(key, config, batch)


def strategy_apply(strategy: Strategy, features: Array, config: StrategyConfig, training: bool) -> tuple[Array, Strategy]:
    """Scan the semi-recurrent policy; inputs are parameters/features/mode, outputs holdings/new BN state."""

    batch, n_time, n_features = features.shape
    n_assets = strategy.b3.shape[-1]
    chex.assert_shape(features, (batch, n_time, n_features))
    features_t = jnp.swapaxes(features, 0, 1)
    chex.assert_shape(features_t, (n_time, batch, n_features))
    xs = (features_t, strategy.w1, strategy.b1, strategy.w2, strategy.b2,
          strategy.w3, strategy.b3, strategy.gamma1, strategy.beta1, strategy.gamma2,
          strategy.beta2, strategy.mean1, strategy.var1, strategy.mean2, strategy.var2)

    def step(previous: Array, values: tuple[Array, ...]):
        """Apply one date-specific network; inputs previous holdings/arrays, outputs holding/BN statistics."""

        (feature, w1, b1, w2, b2, w3, b3, gamma1, beta1, gamma2, beta2,
         running_mean1, running_var1, running_mean2, running_var2) = values
        chex.assert_shape(feature, (batch, n_features))
        network_input = jnp.concatenate((feature, previous), axis=-1)
        chex.assert_shape(network_input, (batch, n_features + n_assets))
        h1 = network_input @ w1.T + b1
        hidden = h1.shape[-1]
        chex.assert_shape(h1, (batch, hidden))
        mean1 = jnp.mean(h1, axis=0) if training else running_mean1
        var1 = jnp.var(h1, axis=0) if training else running_var1
        chex.assert_shape(mean1, (hidden,))
        chex.assert_shape(var1, (hidden,))
        normalized1 = (h1 - mean1) / jnp.sqrt(var1 + config.bn_epsilon)
        chex.assert_shape(normalized1, (batch, hidden))
        affine1 = gamma1 * normalized1 + beta1
        chex.assert_shape(affine1, (batch, hidden))
        h1 = jax.nn.relu(affine1)
        chex.assert_shape(h1, (batch, hidden))
        h2 = h1 @ w2.T + b2
        chex.assert_shape(h2, (batch, hidden))
        mean2 = jnp.mean(h2, axis=0) if training else running_mean2
        var2 = jnp.var(h2, axis=0) if training else running_var2
        chex.assert_shape(mean2, (hidden,))
        chex.assert_shape(var2, (hidden,))
        normalized2 = (h2 - mean2) / jnp.sqrt(var2 + config.bn_epsilon)
        chex.assert_shape(normalized2, (batch, hidden))
        affine2 = gamma2 * normalized2 + beta2
        chex.assert_shape(affine2, (batch, hidden))
        h2 = jax.nn.relu(affine2)
        chex.assert_shape(h2, (batch, hidden))
        holding = h2 @ w3.T + b3
        chex.assert_shape(holding, (batch, n_assets))
        if config.stock_only and n_assets == 2:
            holding = holding.at[:, 1].set(0.0)
            chex.assert_shape(holding, (batch, n_assets))
        updated_mean1 = config.bn_momentum * running_mean1 + (1.0 - config.bn_momentum) * mean1
        chex.assert_shape(updated_mean1, (hidden,))
        updated_var1 = config.bn_momentum * running_var1 + (1.0 - config.bn_momentum) * var1
        chex.assert_shape(updated_var1, (hidden,))
        updated_mean2 = config.bn_momentum * running_mean2 + (1.0 - config.bn_momentum) * mean2
        chex.assert_shape(updated_mean2, (hidden,))
        updated_var2 = config.bn_momentum * running_var2 + (1.0 - config.bn_momentum) * var2
        chex.assert_shape(updated_var2, (hidden,))
        updated = (updated_mean1, updated_var1, updated_mean2, updated_var2)
        return holding, (holding, *updated)

    initial = jnp.zeros((batch, n_assets), dtype=features.dtype)
    chex.assert_shape(initial, (batch, n_assets))
    _, outputs = jax.lax.scan(step, initial, xs)
    holdings_t, mean1, var1, mean2, var2 = outputs
    chex.assert_shape(holdings_t, (n_time, batch, n_assets))
    holdings = jnp.swapaxes(holdings_t, 0, 1)
    chex.assert_shape(holdings, (batch, n_time, n_assets))
    new_strategy = eqx.tree_at(lambda s: (s.mean1, s.var1, s.mean2, s.var2), strategy,
                               (mean1, var1, mean2, var2))
    return holdings, new_strategy


def hedging_loss(strategy: Strategy, prices: Array, features: Array, epsilon: float,
                 strategy_config: StrategyConfig, training: bool, alpha: float = 0.95,
                 strike: float = 100.0) -> tuple[Array, tuple[Array, ...], Strategy]:
    """Compute CVaR and accounting; inputs policy/paths/cost/mode, outputs loss, diagnostics, BN state."""

    batch, n_points, n_assets = prices.shape
    n_time = n_points - 1
    chex.assert_shape(features, (batch, n_time, features.shape[-1]))
    holdings, updated = strategy_apply(strategy, features, strategy_config, training)
    chex.assert_shape(holdings, (batch, n_time, n_assets))

    def accounting_step(carry: tuple[Array, Array, Array], values: tuple[Array, Array, Array]):
        """Accumulate one trading date; inputs are prior holding/totals and prices, outputs new totals/trade."""

        previous, gains, costs = carry
        holding, current_price, next_price = values
        chex.assert_shape(previous, (batch, n_assets))
        price_change = next_price - current_price
        chex.assert_shape(price_change, (batch, n_assets))
        gain_increment = jnp.sum(holding * price_change, axis=1)
        chex.assert_shape(gain_increment, (batch,))
        trade = holding - previous
        chex.assert_shape(trade, (batch, n_assets))
        cost_increment = epsilon * jnp.sum(jnp.abs(trade) * current_price, axis=1)
        chex.assert_shape(cost_increment, (batch,))
        next_gains = gains + gain_increment
        chex.assert_shape(next_gains, (batch,))
        next_costs = costs + cost_increment
        chex.assert_shape(next_costs, (batch,))
        return (holding, next_gains, next_costs), trade

    initial_holding = jnp.zeros((batch, n_assets), dtype=holdings.dtype)
    chex.assert_shape(initial_holding, (batch, n_assets))
    initial_total = jnp.zeros((batch,), dtype=holdings.dtype)
    chex.assert_shape(initial_total, (batch,))
    holdings_t = jnp.swapaxes(holdings, 0, 1)
    chex.assert_shape(holdings_t, (n_time, batch, n_assets))
    current_prices_t = jnp.swapaxes(prices[:, :-1], 0, 1)
    chex.assert_shape(current_prices_t, (n_time, batch, n_assets))
    next_prices_t = jnp.swapaxes(prices[:, 1:], 0, 1)
    chex.assert_shape(next_prices_t, (n_time, batch, n_assets))
    accounting_inputs = (holdings_t, current_prices_t, next_prices_t)
    (_, gains, costs), trades_t = jax.lax.scan(
        accounting_step, (initial_holding, initial_total, initial_total), accounting_inputs)
    chex.assert_shape(gains, (batch,))
    chex.assert_shape(costs, (batch,))
    chex.assert_shape(trades_t, (n_time, batch, n_assets))
    trades = jnp.swapaxes(trades_t, 0, 1)
    chex.assert_shape(trades, (batch, n_time, n_assets))
    payoff = jnp.maximum(prices[:, -1, 0] - strike, 0.0)
    chex.assert_shape(payoff, (batch,))
    error = strategy.cash + gains - costs - payoff
    chex.assert_shape(error, (batch,))
    losses = -error
    chex.assert_shape(losses, (batch,))
    loss = strategy.q + jnp.mean(jnp.maximum(losses - strategy.q, 0.0)) / (1.0 - alpha)
    chex.assert_shape(loss, ())
    turnover = jnp.sum(jnp.abs(trades), axis=(1, 2))
    chex.assert_shape(turnover, (batch,))
    return loss, (error, gains, costs, turnover, holdings), updated


def make_steps(optimizer: optax.GradientTransformation, strategy_config: StrategyConfig,
               epsilon: float, alpha: float = 0.95, strike: float = 100.0):
    """Create top-level jitted train/eval callables; inputs optimizer/config/cost, outputs compiled steps."""

    @eqx.filter_jit
    def train_step(strategy: Strategy, opt_state: optax.OptState, prices: Array, features: Array):
        """Run one compiled Adam step; inputs state and batch, outputs updated state/loss/gradient norm."""

        def objective(candidate: Strategy):
            """Differentiate training CVaR; input candidate strategy, output loss and updated BN strategy."""

            loss, _, updated = hedging_loss(candidate, prices, features, epsilon, strategy_config, True, alpha, strike)
            return loss, updated

        (loss, bn_strategy), grads = eqx.filter_value_and_grad(objective, has_aux=True)(strategy)
        # cash is the post-hoc indifference price, not a training parameter under CVaR
        grads = eqx.tree_at(lambda g: g.cash, grads, jnp.zeros_like(grads.cash))
        updates, opt_state = optimizer.update(grads, opt_state, strategy)
        strategy = eqx.apply_updates(strategy, updates)
        strategy = eqx.tree_at(lambda s: (s.mean1, s.var1, s.mean2, s.var2), strategy,
                               (bn_strategy.mean1, bn_strategy.var1, bn_strategy.mean2, bn_strategy.var2))
        grad_norm = optax.tree.norm(grads)
        chex.assert_shape(loss, ())
        chex.assert_shape(grad_norm, ())
        return strategy, opt_state, loss, grad_norm

    @eqx.filter_jit
    def evaluate(strategy: Strategy, prices: Array, features: Array):
        """Evaluate frozen-BN policy; inputs strategy/paths, outputs loss and pathwise diagnostics."""

        loss, diagnostics, _ = hedging_loss(strategy, prices, features, epsilon, strategy_config, False, alpha, strike)
        return loss, diagnostics

    return train_step, evaluate


def train(config: Config) -> tuple[Strategy, dict[str, list[float]]]:
    """Train with online simulation and fixed validation; input config, outputs best strategy/history."""

    key = jax.random.PRNGKey(config.training.seed)
    key, init_key, validation_key = jax.random.split(key, 3)
    strategy = initialize_strategy(init_key, config.market)
    optimizer = optax.adam(config.training.learning_rate)
    opt_state = optimizer.init(eqx.filter(strategy, eqx.is_array))
    train_step, evaluate = make_steps(optimizer, config.strategy, config.epsilon, config.alpha, config.market.strike)
    simulate_training = jax.jit(lambda simulation_key: simulate_market(
        simulation_key, config.market, config.training.batch_size))
    simulate_validation = jax.jit(lambda simulation_key: simulate_market(
        simulation_key, config.market, config.training.validation_size))
    validation_prices, validation_features, _ = simulate_validation(validation_key)
    calibration_config = dataclasses.replace(config.strategy, bn_momentum=0.0)

    @eqx.filter_jit
    def calibrate(candidate: Strategy) -> Strategy:
        """Recompute exact fixed-validation BN statistics; input strategy, output calibrated strategy."""

        _, calibrated = strategy_apply(candidate, validation_features, calibration_config, True)
        return calibrated

    best_strategy = strategy
    best_loss = float("inf")
    history: dict[str, list[float]] = {"step": [], "train_loss": [], "validation_loss": [], "grad_norm": []}
    for step in range(config.training.steps):
        key, batch_key = jax.random.split(key)
        prices, features, _ = simulate_training(batch_key)
        strategy, opt_state, loss, grad_norm = train_step(strategy, opt_state, prices, features)
        if step % config.training.validation_frequency == 0 or step == config.training.steps - 1:
            strategy = calibrate(strategy)
            validation_loss, _ = evaluate(strategy, validation_prices, validation_features)
            value = float(validation_loss)
            history["step"].append(step)
            history["train_loss"].append(float(loss))
            history["validation_loss"].append(value)
            history["grad_norm"].append(float(grad_norm))
            if value < best_loss:
                best_loss = value
                best_strategy = jax.tree.map(lambda x: jnp.array(x), strategy)
            print(f"step={step:6d} train={float(loss):.6f} validation={value:.6f} cash={float(strategy.cash):.4f}")
    return best_strategy, history


def evaluate_chunks(strategy: Strategy, config: Config) -> dict[str, float | list[float]]:
    """Evaluate independent chunks; inputs trained strategy/config, output aggregate metrics dictionary."""

    paths = 1_000_000 if config.evaluation.paper_scale else config.evaluation.paths
    _, evaluate = make_steps(optax.adam(config.training.learning_rate), config.strategy,
                             config.epsilon, config.alpha, config.market.strike)
    simulate_full_chunk = jax.jit(lambda simulation_key: simulate_market(
        simulation_key, config.market, config.evaluation.chunk_size))
    key = jax.random.PRNGKey(config.training.seed + 10_000)
    errors, costs, turnovers, holdings, turnover_dates = [], [], [], [], []
    chunk_sizes = []
    remaining = paths
    while remaining:
        size = min(remaining, config.evaluation.chunk_size)
        key, chunk_key = jax.random.split(key)
        if size == config.evaluation.chunk_size:
            prices, features, _ = simulate_full_chunk(chunk_key)
        else:
            simulate_remainder = jax.jit(lambda simulation_key: simulate_market(
                simulation_key, config.market, size))
            prices, features, _ = simulate_remainder(chunk_key)
        _, diagnostics = evaluate(strategy, prices, features)
        error, _, cost, turnover, chunk_holdings = diagnostics
        errors.append(np.asarray(error))
        costs.append(np.asarray(cost))
        turnovers.append(np.asarray(turnover))
        holdings.append(np.asarray(chunk_holdings).mean(axis=0))
        previous = np.concatenate((np.zeros_like(np.asarray(chunk_holdings)[:, :1]),
                                   np.asarray(chunk_holdings)[:, :-1]), axis=1)
        turnover_dates.append(np.abs(np.asarray(chunk_holdings) - previous).sum(axis=2).mean(axis=0))
        chunk_sizes.append(size)
        remaining -= size
    error_np = np.concatenate(errors)
    cost_np = np.concatenate(costs)
    turnover_np = np.concatenate(turnovers)
    losses_np = -error_np
    threshold = float(np.quantile(losses_np, config.alpha))
    cvar = float(np.mean(losses_np[losses_np >= threshold]))
    return {"mse": float(np.mean(error_np**2)), "cvar": cvar, "cash": float(strategy.cash),
            "error_mean": float(np.mean(error_np)), "error_std": float(np.std(error_np)),
            "error_quantiles": np.quantile(error_np, [0.01, 0.05, 0.5, 0.95, 0.99]).tolist(),
            "mean_cost": float(np.mean(cost_np)), "mean_turnover": float(np.mean(turnover_np)),
            "mean_holdings": np.average(np.stack(holdings), axis=0, weights=chunk_sizes).tolist(),
            "mean_turnover_by_date": np.average(np.stack(turnover_dates), axis=0,
                                                 weights=chunk_sizes).tolist(),
            "errors": error_np.tolist()}


def save_artifacts(strategy: Strategy, history: dict[str, list[float]], metrics: dict, config: Config) -> Path:
    """Save config/model/metrics/plots; inputs completed run objects, output artifact directory path."""

    directory = Path(config.evaluation.output_dir) / f"{config.market.name}_cost_{config.epsilon:g}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2))
    diagnostic_prices = diagnostic_learned = diagnostic_analytic = None
    if config.market.name == "black_scholes" and config.epsilon == 0.0:
        diagnostic_prices, diagnostic_features = simulate_black_scholes(
            jax.random.PRNGKey(config.training.seed + 20_000), config.market, 4_096)
        diagnostic_learned, _ = strategy_apply(strategy, diagnostic_features, config.strategy, False)
        times = jnp.arange(config.market.n_time, dtype=strategy.w1.dtype) * (config.market.horizon / config.market.n_time)
        chex.assert_shape(times, (config.market.n_time,))
        tau = config.market.horizon - times
        chex.assert_shape(tau, (config.market.n_time,))
        diagnostic_spots = diagnostic_prices[:, :-1, 0]
        chex.assert_shape(diagnostic_spots, (4_096, config.market.n_time))
        d1 = (jnp.log(diagnostic_spots / config.market.strike) + 0.5 * config.market.volatility**2 * tau[None]) / (config.market.volatility * jnp.sqrt(tau[None]))
        chex.assert_shape(d1, (4_096, config.market.n_time))
        diagnostic_analytic = 0.5 * (1.0 + jax.lax.erf(d1 / jnp.sqrt(2.0)))
        chex.assert_shape(diagnostic_analytic, (4_096, config.market.n_time))
        difference = diagnostic_learned[:, :, 0] - diagnostic_analytic
        chex.assert_shape(difference, (4_096, config.market.n_time))
        delta_mse = jnp.mean(jnp.square(difference))
        chex.assert_shape(delta_mse, ())
        delta_rmse = jnp.sqrt(delta_mse)
        chex.assert_shape(delta_rmse, ())
        delta_correlation_matrix = jnp.corrcoef(
            diagnostic_learned[:, :, 0].ravel(), diagnostic_analytic.ravel())
        chex.assert_shape(delta_correlation_matrix, (2, 2))
        delta_correlation = delta_correlation_matrix[0, 1]
        chex.assert_shape(delta_correlation, ())
        metrics["analytic_delta_rmse"] = float(delta_rmse)
        metrics["analytic_delta_correlation"] = float(delta_correlation)
    serializable_metrics = {key: value for key, value in metrics.items() if key != "errors"}
    (directory / "metrics.json").write_text(json.dumps(serializable_metrics, indent=2))
    (directory / "history.json").write_text(json.dumps(history, indent=2))
    eqx.tree_serialise_leaves(directory / "model.eqx", strategy)
    figure, axis = plt.subplots()
    axis.plot(history["step"], history["train_loss"], label="train")
    axis.plot(history["step"], history["validation_loss"], label="validation")
    axis.set_yscale("log")
    axis.legend()
    figure.savefig(directory / "loss.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots()
    axis.hist(metrics["errors"], bins=100)
    axis.set_xlabel("terminal hedging error")
    figure.savefig(directory / "hedging_error.png", dpi=160)
    plt.close(figure)
    mean_holdings = np.asarray(metrics["mean_holdings"])
    figure, axis = plt.subplots()
    axis.plot(mean_holdings)
    axis.set_xlabel("trading date")
    axis.set_ylabel("mean holding")
    figure.savefig(directory / "holdings.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots()
    axis.plot(metrics["mean_turnover_by_date"])
    axis.set_xlabel("trading date")
    axis.set_ylabel("mean absolute turnover")
    figure.savefig(directory / "turnover.png", dpi=160)
    plt.close(figure)
    if config.market.name == "black_scholes" and config.epsilon == 0.0:
        figure, axis = plt.subplots()
        for date in sorted({0, config.market.n_time // 2, config.market.n_time - 1}):
            spot_values = np.asarray(diagnostic_prices[:, date, 0])
            order = np.argsort(spot_values)
            axis.plot(spot_values[order], np.asarray(diagnostic_learned[order, date, 0]), label=f"learned t={date}")
            axis.plot(spot_values[order], np.asarray(diagnostic_analytic[order, date]), "--", label=f"analytic t={date}")
        axis.set_xlabel("spot on simulated histories")
        axis.set_ylabel("stock holding / delta")
        axis.legend(fontsize="small")
        figure.savefig(directory / "analytic_delta.png", dpi=160)
        plt.close(figure)
    comparison = Path(config.evaluation.output_dir) / "comparison.csv"
    rows = {}
    if comparison.exists():
        for line in comparison.read_text().splitlines()[1:]:
            fields = line.split(",")
            if len(fields) == 6:
                fields = [fields[0], fields[1], "0", "float32", "100000", *fields[2:]]
            rows[(fields[0], fields[1])] = fields
    evaluation_paths = 1_000_000 if config.evaluation.paper_scale else config.evaluation.paths
    row = [config.market.name, f"{config.epsilon:g}", str(config.training.seed), config.market.dtype,
           str(evaluation_paths), f"{config.alpha:g}", str(serializable_metrics["mse"]),
           str(serializable_metrics["cvar"]), str(serializable_metrics["cash"]),
           str(serializable_metrics["mean_cost"]), str(serializable_metrics["mean_turnover"])]
    rows[(row[0], row[1])] = row
    header = "market,epsilon,seed,dtype,evaluation_paths,alpha,mse,cvar,cash,mean_cost,mean_turnover\n"
    comparison.write_text(header + "\n".join(",".join(value) for value in rows.values()) + "\n")
    return directory


def main(config: Config) -> None:
    """Run a CLI experiment; input is Tyro Config and output consists of printed/saved artifacts."""

    if config.market.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    strategy, history = train(config)
    metrics = evaluate_chunks(strategy, config)
    metrics["final_train_cvar"] = history["train_loss"][-1]
    metrics["best_validation_cvar"] = min(history["validation_loss"])
    # CVaR indifference price: cash such that CVaR(losses - cash) = 0
    strategy = eqx.tree_at(lambda s: s.cash, strategy, jnp.asarray(metrics["cvar"], dtype=strategy.cash.dtype))
    metrics["cash"] = metrics["cvar"]
    directory = save_artifacts(strategy, history, metrics, config)
    print(json.dumps({key: value for key, value in metrics.items() if key != "errors"}, indent=2))
    print(f"saved={directory}")


if __name__ == "__main__":
    main(tyro.cli(Config))

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

from wdro import (
    RobustConfig,
    RobustState,
    convex_order_penalty,
    entropic_dual,
    joint_cvar_dual,
    per_sample_cvar_dual,
    plain_average_dual,
    plain_cvar_of_average_dual,
    relaxed_kl_cvar_dual,
    quadratic_transport_cost,
    sample_langevin_proposal,
)

Array = jax.Array


@chex.dataclass
class MarketConfig:
    """Configure the simulated market; all fields are scalar model inputs."""

    name: Literal["black_scholes", "heston", "lsv"] = "black_scholes"
    spot: float = 100.0
    strike: float = 100.0
    volatility: float = 0.2
    kappa: float = 3.0
    long_variance: float = 0.04
    vol_of_variance: float = 0.4
    correlation: float = -0.7
    initial_variance: float = 0.04
    n_time: int = 30
    horizon: float = 30.0 / 365.0
    dtype: Literal["float32", "float64"] = "float32"
    # LSV (Heston + leverage) fields; unused unless name == "lsv".
    leverage_beta: float = -0.3
    leverage_kappa: float = 0.5
    leverage_cap: float = 5.0
    n_substeps: int = 10


@chex.dataclass
class StrategyConfig:
    """Configure paper-style networks; inputs are BN constants and output is architecture metadata."""

    bn_momentum: float = 0.99
    bn_epsilon: float = 1e-5
    stock_only: bool = False
    latent_dim: int = 0


@chex.dataclass
class TrainingConfig:
    """Configure optimization; fields control Adam, online batches, validation, and randomness."""

    steps: int = 20_000
    batch_size: int = 256
    learning_rate: float = 0.005
    validation_size: int = 65_536
    validation_frequency: int = 250
    seed: int = 0
    grad_clip_norm: float | None = None


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
    robust: RobustConfig = dataclasses.field(default_factory=RobustConfig)
    epsilon: float = 0.0
    alpha: float = 0.95


class MLPCore(eqx.Module):
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


class GRUCore(eqx.Module):
    """Hold time-specific GRU-cell + LayerNorm + output-head parameters as leading-time arrays."""

    w_ir: Array
    w_iz: Array
    w_in: Array
    w_hr: Array
    w_hz: Array
    w_hn: Array
    b_ir: Array
    b_iz: Array
    b_in: Array
    b_hr: Array
    b_hz: Array
    b_hn: Array
    ln_gamma: Array
    ln_beta: Array
    w_out: Array
    b_out: Array


class Strategy(eqx.Module):
    """Hold the recurrent policy core (MLP+BN or GRU+LayerNorm) plus the scalar training extras."""

    core: MLPCore | GRUCore
    cash: Array
    q: Array


def market_dimensions(config: MarketConfig) -> tuple[int, int]:
    """Return (number of tradable assets, number of observable features) for a market config."""

    return (1, 1) if config.name == "black_scholes" else (2, 2)


def _initialize_mlp_core(key: Array, n_assets: int, n_features: int, n_time: int, dtype) -> MLPCore:
    """Initialize paper-width, time-specific BN+MLPs; inputs are key/dims, output is an MLPCore."""

    n_input = n_features + n_assets
    hidden = n_assets + 15
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
    return MLPCore(w1, b1, w2, b2, w3, b3, ones1, zeros1, ones1, zeros1, zeros1, ones1, zeros1, ones1)


def _initialize_gru_core(key: Array, n_assets: int, n_features: int, n_time: int, latent_dim: int, dtype) -> GRUCore:
    """Initialize a per-timestep GRU cell + LayerNorm + output head; inputs key/dims, output a GRUCore."""

    hidden = n_assets + latent_dim
    keys = jax.random.split(key, 8 * n_time).reshape(8, n_time, 2)
    chex.assert_shape(keys, (8, n_time, 2))
    scale_in = jnp.asarray(1.0 / np.sqrt(float(n_features)), dtype=dtype)
    chex.assert_shape(scale_in, ())
    scale_hidden = jnp.asarray(1.0 / np.sqrt(float(hidden)), dtype=dtype)
    chex.assert_shape(scale_hidden, ())

    def input_weight(date_key):
        return jax.random.uniform(date_key, (hidden, n_features), minval=-scale_in, maxval=scale_in, dtype=dtype)

    def hidden_weight(date_key):
        return jax.random.uniform(date_key, (hidden, hidden), minval=-scale_hidden, maxval=scale_hidden, dtype=dtype)

    def hidden_bias(date_key):
        return jax.random.uniform(date_key, (hidden,), minval=-scale_hidden, maxval=scale_hidden, dtype=dtype)

    w_ir = jax.vmap(input_weight)(keys[0])
    chex.assert_shape(w_ir, (n_time, hidden, n_features))
    w_iz = jax.vmap(input_weight)(keys[1])
    w_in = jax.vmap(input_weight)(keys[2])
    w_hr = jax.vmap(hidden_weight)(keys[3])
    chex.assert_shape(w_hr, (n_time, hidden, hidden))
    w_hz = jax.vmap(hidden_weight)(keys[4])
    w_hn = jax.vmap(hidden_weight)(keys[5])
    b_ir = jax.vmap(hidden_bias)(keys[6])
    chex.assert_shape(b_ir, (n_time, hidden))
    b_iz = jnp.zeros((n_time, hidden), dtype=dtype)
    b_in = jnp.zeros((n_time, hidden), dtype=dtype)
    b_hr = jnp.zeros((n_time, hidden), dtype=dtype)
    b_hz = jnp.zeros((n_time, hidden), dtype=dtype)
    b_hn = jnp.zeros((n_time, hidden), dtype=dtype)
    ln_gamma = jnp.ones((n_time, hidden), dtype=dtype)
    ln_beta = jnp.zeros((n_time, hidden), dtype=dtype)
    w_out = jax.vmap(lambda date_key: jax.random.uniform(
        date_key, (n_assets, hidden), minval=-scale_hidden, maxval=scale_hidden, dtype=dtype))(keys[7])
    chex.assert_shape(w_out, (n_time, n_assets, hidden))
    b_out = jnp.zeros((n_time, n_assets), dtype=dtype)
    return GRUCore(w_ir, w_iz, w_in, w_hr, w_hz, w_hn, b_ir, b_iz, b_in, b_hr, b_hz, b_hn,
                   ln_gamma, ln_beta, w_out, b_out)


def initialize_strategy(key: Array, market: MarketConfig, strategy_config: StrategyConfig) -> Strategy:
    """Initialize the recurrent policy; inputs are a PRNG key/market/strategy config, output is Strategy."""

    n_assets, n_features = market_dimensions(market)
    n_time = market.n_time
    dtype = jnp.float64 if market.dtype == "float64" else jnp.float32
    if strategy_config.latent_dim > 0:
        core = _initialize_gru_core(key, n_assets, n_features, n_time, strategy_config.latent_dim, dtype)
    else:
        core = _initialize_mlp_core(key, n_assets, n_features, n_time, dtype)
    cash = jnp.asarray(0.0, dtype=dtype)
    chex.assert_shape(cash, ())
    q = jnp.asarray(0.0, dtype=dtype)
    chex.assert_shape(q, ())
    return Strategy(core=core, cash=cash, q=q)


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


def simulate_lsv(key: Array, config: MarketConfig, batch: int) -> tuple[Array, Array, Array]:
    """Simulate Heston + leverage on a fine sub-grid; outputs tradables, features, variance at the coarse grid."""

    dtype = jnp.float64 if config.dtype == "float64" else jnp.float32
    n_fine = config.n_time * config.n_substeps
    dt = jnp.asarray(config.horizon / n_fine, dtype=dtype)
    chex.assert_shape(dt, ())
    keys = jax.random.split(key, n_fine * 2).reshape(n_fine, 2, 2)
    chex.assert_shape(keys, (n_fine, 2, 2))
    kappa = jnp.asarray(config.kappa, dtype=dtype)
    chex.assert_shape(kappa, ())
    theta = jnp.asarray(config.long_variance, dtype=dtype)
    chex.assert_shape(theta, ())
    xi = jnp.asarray(config.vol_of_variance, dtype=dtype)
    chex.assert_shape(xi, ())
    rho = jnp.asarray(config.correlation, dtype=dtype)
    chex.assert_shape(rho, ())
    spot0 = jnp.asarray(config.spot, dtype=dtype)
    chex.assert_shape(spot0, ())
    beta = jnp.asarray(config.leverage_beta, dtype=dtype)
    chex.assert_shape(beta, ())
    kappa_l = jnp.asarray(config.leverage_kappa, dtype=dtype)
    chex.assert_shape(kappa_l, ())
    leverage_cap = jnp.asarray(config.leverage_cap, dtype=dtype)
    chex.assert_shape(leverage_cap, ())
    decay = jnp.exp(-kappa * dt)
    chex.assert_shape(decay, ())
    c = xi**2 * (1.0 - decay) / (4.0 * kappa)
    chex.assert_shape(c, ())
    dof = 4.0 * kappa * theta / xi**2
    chex.assert_shape(dof, ())
    fine_times = jnp.arange(n_fine, dtype=dtype) * dt
    chex.assert_shape(fine_times, (n_fine,))

    def step(carry: tuple[Array, Array, Array], values: tuple[Array, Array]):
        """Advance one fine sub-interval; variance leg exact CIR, stock leg Euler with left-endpoint leverage."""

        variance, log_stock, integrated = carry
        step_keys, t_left = values
        chex.assert_shape(variance, (batch,))
        chex.assert_shape(t_left, ())
        noncentrality = decay * variance / c
        chex.assert_shape(noncentrality, (batch,))
        poisson_key, gamma_key = jax.random.split(step_keys[0])
        counts = jax.random.poisson(poisson_key, noncentrality / 2.0, shape=(batch,))
        chex.assert_shape(counts, (batch,))
        chi_square = 2.0 * jax.random.gamma(gamma_key, dof / 2.0 + counts, dtype=dtype)
        chex.assert_shape(chi_square, (batch,))
        next_variance = c * chi_square
        chex.assert_shape(next_variance, (batch,))
        # Piecewise-constant integrated variance over this fine sub-interval (left endpoint).
        interval_variance = variance * dt
        chex.assert_shape(interval_variance, (batch,))
        z = jax.random.normal(step_keys[1], (batch,), dtype=dtype)
        chex.assert_shape(z, (batch,))
        # Exact identity rho * integral(sqrt(V) dW^V) over the sub-interval, independent of
        # the stock leg's dynamics (a property of the CIR variance process alone), so it is
        # reused unmodified from the plain-Heston scheme and only rescaled by leverage below.
        correlated = (rho / xi) * (next_variance - variance - kappa * (theta * dt - interval_variance))
        chex.assert_shape(correlated, (batch,))
        raw_leverage = (jnp.exp(log_stock) / spot0) ** beta * (1.0 + kappa_l * t_left)
        chex.assert_shape(raw_leverage, (batch,))
        # Cap the leverage multiplier: uncapped, beta<0 blows up as S->0 in the left
        # tail, feeding a runaway drift/diffusion spiral (verified empirically at
        # n_time=100, n_substeps=10 with the default in-dist params). Capping the
        # multiplier (not the spot ratio) is standard practice in local-vol/SLV
        # simulation for boundary stability.
        leverage = jnp.clip(raw_leverage, 0.0, leverage_cap)
        chex.assert_shape(leverage, (batch,))
        next_log_stock = (log_stock - 0.5 * leverage**2 * interval_variance
                          + leverage * correlated
                          + leverage * jnp.sqrt(1.0 - rho**2) * jnp.sqrt(jnp.maximum(interval_variance, 0.0)) * z)
        chex.assert_shape(next_log_stock, (batch,))
        next_integrated = integrated + interval_variance
        chex.assert_shape(next_integrated, (batch,))
        return (next_variance, next_log_stock, next_integrated), (next_variance, next_log_stock, next_integrated)

    initial = (jnp.full((batch,), config.initial_variance, dtype=dtype),
               jnp.full((batch,), jnp.log(config.spot), dtype=dtype),
               jnp.zeros((batch,), dtype=dtype))
    (_, _, _), (variance_fine, log_fine, integrated_fine) = jax.lax.scan(step, initial, (keys, fine_times))
    chex.assert_shape(variance_fine, (n_fine, batch))
    variance_fine = jnp.concatenate((initial[0][None], variance_fine), axis=0)
    log_fine = jnp.concatenate((initial[1][None], log_fine), axis=0)
    integrated_fine = jnp.concatenate((initial[2][None], integrated_fine), axis=0)
    chex.assert_shape(variance_fine, (n_fine + 1, batch))
    # Sub-sample fine states down to the coarse hedging grid; the strategy only trades here.
    coarse_index = jnp.arange(config.n_time + 1) * config.n_substeps
    chex.assert_shape(coarse_index, (config.n_time + 1,))
    variance = variance_fine[coarse_index].T
    chex.assert_shape(variance, (batch, config.n_time + 1))
    logs = log_fine[coarse_index].T
    chex.assert_shape(logs, (batch, config.n_time + 1))
    integrated = integrated_fine[coarse_index].T
    chex.assert_shape(integrated, (batch, config.n_time + 1))
    times = jnp.linspace(0.0, config.horizon, config.n_time + 1, dtype=dtype)
    chex.assert_shape(times, (config.n_time + 1,))
    remaining = config.horizon - times[None, :]
    chex.assert_shape(remaining, (1, config.n_time + 1))
    # Conditional expectation of remaining integrated variance depends only on the variance
    # marginal (unaffected by leverage), so this closed form is reused unmodified from Heston.
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
    if config.name == "lsv":
        return simulate_lsv(key, config, batch)
    return simulate_heston(key, config, batch)


def encode_trajectory(prices: Array, variance: Array | None, config: MarketConfig) -> Array:
    """Encode valid market paths as Euclidean coordinates; inputs are paths/variance/config, output is coordinates."""

    batch, n_points, n_assets = prices.shape
    n_time = n_points - 1
    expected_assets, _ = market_dimensions(config)
    chex.assert_shape(prices, (batch, n_time + 1, expected_assets))
    log_stock = jnp.log(prices[:, 1:, 0])
    chex.assert_shape(log_stock, (batch, n_time))
    if config.name == "black_scholes":
        coordinates = log_stock
        chex.assert_shape(coordinates, (batch, n_time))
        return coordinates
    if variance is None:
        raise ValueError("Heston trajectory encoding requires variance paths")
    chex.assert_shape(variance, (batch, n_time + 1))
    variance_floor = jnp.asarray(1e-12, dtype=variance.dtype)
    chex.assert_shape(variance_floor, ())
    positive_variance = jnp.maximum(variance[:, 1:], variance_floor)
    chex.assert_shape(positive_variance, (batch, n_time))
    log_variance = jnp.log(positive_variance)
    chex.assert_shape(log_variance, (batch, n_time))
    coordinates = jnp.concatenate((log_stock, log_variance), axis=1)
    chex.assert_shape(coordinates, (batch, 2 * n_time))
    return coordinates


def decode_trajectory(coordinates: Array, config: MarketConfig) -> tuple[Array, Array, Array | None]:
    """Decode Euclidean coordinates into valid market paths; inputs are coordinates/config, outputs paths/features/variance."""

    batch, dimension = coordinates.shape
    n_time = config.n_time
    dtype = coordinates.dtype
    log_initial = jnp.full((batch, 1), jnp.log(config.spot), dtype=dtype)
    chex.assert_shape(log_initial, (batch, 1))
    if config.name == "black_scholes":
        chex.assert_shape(coordinates, (batch, n_time))
        logs = jnp.concatenate((log_initial, coordinates), axis=1)
        chex.assert_shape(logs, (batch, n_time + 1))
        stock = jnp.exp(logs)[:, :, None]
        chex.assert_shape(stock, (batch, n_time + 1, 1))
        features = logs[:, :-1, None]
        chex.assert_shape(features, (batch, n_time, 1))
        return stock, features, None
    chex.assert_shape(coordinates, (batch, 2 * n_time))
    log_stock_tail = coordinates[:, :n_time]
    chex.assert_shape(log_stock_tail, (batch, n_time))
    log_variance_tail = coordinates[:, n_time:]
    chex.assert_shape(log_variance_tail, (batch, n_time))
    logs = jnp.concatenate((log_initial, log_stock_tail), axis=1)
    chex.assert_shape(logs, (batch, n_time + 1))
    initial_variance = jnp.full((batch, 1), config.initial_variance, dtype=dtype)
    chex.assert_shape(initial_variance, (batch, 1))
    variance_tail = jnp.exp(log_variance_tail)
    chex.assert_shape(variance_tail, (batch, n_time))
    variance = jnp.concatenate((initial_variance, variance_tail), axis=1)
    chex.assert_shape(variance, (batch, n_time + 1))
    dt = jnp.asarray(config.horizon / n_time, dtype=dtype)
    chex.assert_shape(dt, ())
    interval_variance = variance[:, :-1] * dt
    chex.assert_shape(interval_variance, (batch, n_time))
    integrated_tail = jnp.cumsum(interval_variance, axis=1)
    chex.assert_shape(integrated_tail, (batch, n_time))
    integrated = jnp.concatenate((jnp.zeros((batch, 1), dtype=dtype), integrated_tail), axis=1)
    chex.assert_shape(integrated, (batch, n_time + 1))
    times = jnp.linspace(0.0, config.horizon, n_time + 1, dtype=dtype)
    chex.assert_shape(times, (n_time + 1,))
    remaining = config.horizon - times[None, :]
    chex.assert_shape(remaining, (1, n_time + 1))
    adjustment = ((variance - config.long_variance) / config.kappa) * (
        1.0 - jnp.exp(-config.kappa * remaining)) + config.long_variance * remaining
    chex.assert_shape(adjustment, (batch, n_time + 1))
    variance_swap = integrated + adjustment
    chex.assert_shape(variance_swap, (batch, n_time + 1))
    stock = jnp.exp(logs)
    chex.assert_shape(stock, (batch, n_time + 1))
    prices = jnp.stack((stock, variance_swap), axis=-1)
    chex.assert_shape(prices, (batch, n_time + 1, 2))
    features = jnp.stack((logs[:, :-1], variance[:, :-1]), axis=-1)
    chex.assert_shape(features, (batch, n_time, 2))
    return prices, features, variance


def _mlp_apply(core: MLPCore, features: Array, config: StrategyConfig, training: bool) -> tuple[Array, MLPCore]:
    """Scan the semi-recurrent BN+MLP policy; inputs are core/features/mode, outputs holdings/new BN state."""

    batch, n_time, n_features = features.shape
    n_assets = core.b3.shape[-1]
    features_t = jnp.swapaxes(features, 0, 1)
    chex.assert_shape(features_t, (n_time, batch, n_features))
    xs = (features_t, core.w1, core.b1, core.w2, core.b2,
          core.w3, core.b3, core.gamma1, core.beta1, core.gamma2,
          core.beta2, core.mean1, core.var1, core.mean2, core.var2)

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
    new_core = eqx.tree_at(lambda c: (c.mean1, c.var1, c.mean2, c.var2), core,
                           (mean1, var1, mean2, var2))
    return holdings, new_core


def _gru_apply(core: GRUCore, features: Array, config: StrategyConfig) -> Array:
    """Scan a per-timestep GRU cell + LayerNorm policy; inputs are core/features, output is holdings.

    LayerNorm normalizes the hidden state per-sample every step (no batch statistics, no
    train/eval distinction) and the sigmoid-bounded reset/update gates control how much of the
    latent state carries forward -- together these are the standard fix for the vanishing/
    exploding-gradient blowup that BPTT through a long BN+MLP scan hits at large n_time.
    """

    batch, n_time, n_features = features.shape
    hidden = core.w_hr.shape[-1]
    n_assets = core.b_out.shape[-1]
    features_t = jnp.swapaxes(features, 0, 1)
    chex.assert_shape(features_t, (n_time, batch, n_features))
    xs = (features_t, core.w_ir, core.w_iz, core.w_in, core.w_hr, core.w_hz, core.w_hn,
          core.b_ir, core.b_iz, core.b_in, core.b_hr, core.b_hz, core.b_hn,
          core.ln_gamma, core.ln_beta, core.w_out, core.b_out)

    def step(hidden_state: Array, values: tuple[Array, ...]):
        """Apply one date-specific GRU cell; inputs previous hidden/arrays, outputs holding."""

        (feature, w_ir, w_iz, w_in, w_hr, w_hz, w_hn, b_ir, b_iz, b_in, b_hr, b_hz, b_hn,
         ln_gamma, ln_beta, w_out, b_out) = values
        chex.assert_shape(feature, (batch, n_features))
        reset = jax.nn.sigmoid(feature @ w_ir.T + b_ir + hidden_state @ w_hr.T + b_hr)
        chex.assert_shape(reset, (batch, hidden))
        update = jax.nn.sigmoid(feature @ w_iz.T + b_iz + hidden_state @ w_hz.T + b_hz)
        chex.assert_shape(update, (batch, hidden))
        candidate = jnp.tanh(feature @ w_in.T + b_in + reset * (hidden_state @ w_hn.T + b_hn))
        chex.assert_shape(candidate, (batch, hidden))
        raw_next = (1.0 - update) * candidate + update * hidden_state
        chex.assert_shape(raw_next, (batch, hidden))
        mean = jnp.mean(raw_next, axis=-1, keepdims=True)
        variance = jnp.var(raw_next, axis=-1, keepdims=True)
        normalized = (raw_next - mean) / jnp.sqrt(variance + config.bn_epsilon)
        next_hidden = ln_gamma * normalized + ln_beta
        chex.assert_shape(next_hidden, (batch, hidden))
        holding = next_hidden @ w_out.T + b_out
        chex.assert_shape(holding, (batch, n_assets))
        if config.stock_only and n_assets == 2:
            holding = holding.at[:, 1].set(0.0)
            chex.assert_shape(holding, (batch, n_assets))
        return next_hidden, holding

    initial = jnp.zeros((batch, hidden), dtype=features.dtype)
    _, holdings_t = jax.lax.scan(step, initial, xs)
    chex.assert_shape(holdings_t, (n_time, batch, n_assets))
    holdings = jnp.swapaxes(holdings_t, 0, 1)
    chex.assert_shape(holdings, (batch, n_time, n_assets))
    return holdings


def strategy_apply(strategy: Strategy, features: Array, config: StrategyConfig, training: bool) -> tuple[Array, Strategy]:
    """Dispatch to the BN+MLP or GRU+LayerNorm policy; inputs are parameters/features/mode, outputs
    holdings/new strategy (BN running stats updated for MLPCore, unchanged for GRUCore)."""

    if isinstance(strategy.core, GRUCore):
        holdings = _gru_apply(strategy.core, features, config)
        return holdings, strategy
    holdings, new_core = _mlp_apply(strategy.core, features, config, training)
    return holdings, eqx.tree_at(lambda s: s.core, strategy, new_core)


def pathwise_hedging_loss(strategy: Strategy, prices: Array, features: Array, epsilon: float,
                          strategy_config: StrategyConfig, training: bool,
                          strike: float = 100.0) -> tuple[Array, tuple[Array, ...], Strategy]:
    """Compute pathwise hedge losses/accounting; inputs policy/paths/cost/mode, outputs losses/diagnostics/BN state."""

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
    turnover = jnp.sum(jnp.abs(trades), axis=(1, 2))
    chex.assert_shape(turnover, (batch,))
    return losses, (error, gains, costs, turnover, holdings), updated


def hedging_loss(strategy: Strategy, prices: Array, features: Array, epsilon: float,
                 strategy_config: StrategyConfig, training: bool, alpha: float = 0.95,
                 strike: float = 100.0) -> tuple[Array, tuple[Array, ...], Strategy]:
    """Compute nominal CVaR; inputs policy/paths/cost/mode, outputs loss, diagnostics, BN state."""

    losses, diagnostics, updated = pathwise_hedging_loss(
        strategy, prices, features, epsilon, strategy_config, training, strike)
    batch = losses.shape[0]
    chex.assert_shape(losses, (batch,))
    loss = strategy.q + jnp.mean(jnp.maximum(losses - strategy.q, 0.0)) / (1.0 - alpha)
    chex.assert_shape(loss, ())
    return loss, diagnostics, updated


def robust_hedging_loss(
    strategy: Strategy,
    robust_state: RobustState,
    prices: Array,
    variance: Array | None,
    proposal_key: Array,
    market: MarketConfig,
    strategy_config: StrategyConfig,
    robust_config: RobustConfig,
    epsilon: float,
    training: bool,
) -> tuple[Array, tuple[Array, ...], Strategy]:
    """Compute a WDRO hedge objective; inputs are model/base paths/key/configs, outputs objective/diagnostics/BN state."""

    batch, n_points, n_assets = prices.shape
    n_time = n_points - 1
    expected_assets, n_features = market_dimensions(market)
    chex.assert_shape(prices, (batch, n_time + 1, expected_assets))
    raw_anchors = encode_trajectory(prices, variance, market)
    dimension = raw_anchors.shape[1]
    chex.assert_shape(raw_anchors, (batch, dimension))
    chex.assert_shape(robust_state.location, (dimension,))
    chex.assert_shape(robust_state.scale, (dimension,))
    anchors = (raw_anchors - robust_state.location[None, :]) / robust_state.scale[None, :]
    chex.assert_shape(anchors, (batch, dimension))
    lambda_ = jax.nn.softplus(robust_state.raw_lambda)
    chex.assert_shape(lambda_, ())
    reference_scale = jnp.asarray(robust_config.proposal_scale, dtype=anchors.dtype)
    chex.assert_shape(reference_scale, ())
    temperature = robust_config.kl_epsilon + lambda_ * robust_config.tau
    chex.assert_shape(temperature, ())

    def log_target(anchor: Array, sample: Array) -> Array:
        """Evaluate the ULA tilted log target; inputs are one anchor/sample, output is a scalar score."""

        chex.assert_shape(anchor, (dimension,))
        chex.assert_shape(sample, (dimension,))
        physical_sample = sample * robust_state.scale + robust_state.location
        chex.assert_shape(physical_sample, (dimension,))
        proposed_prices, proposed_features, _ = decode_trajectory(physical_sample[None, :], market)
        chex.assert_shape(proposed_prices, (1, n_time + 1, expected_assets))
        chex.assert_shape(proposed_features, (1, n_time, n_features))
        proposed_loss, _, _ = pathwise_hedging_loss(
            strategy, proposed_prices, proposed_features, epsilon, strategy_config, False,
            market.strike)
        chex.assert_shape(proposed_loss, (1,))
        if robust_config.enforce_convex_order:
            order_lambda_ij = jax.nn.softplus(robust_state.raw_lambda_ij)
            order_penalty = convex_order_penalty(
                proposed_prices[..., 0], jnp.asarray(robust_config.convex_order_strikes,
                                                      dtype=proposed_prices.dtype), order_lambda_ij)
            chex.assert_shape(order_penalty, (1,))
            proposed_loss = proposed_loss + order_penalty
            chex.assert_shape(proposed_loss, (1,))
        difference = (sample - anchor) / reference_scale
        chex.assert_shape(difference, (dimension,))
        squared_distance = jnp.sum(jnp.square(difference))
        chex.assert_shape(squared_distance, ())
        if robust_config.cost_power == 1:
            cost = jnp.sqrt(jnp.maximum(squared_distance, 1e-12))
        else:
            cost = 0.5 * squared_distance
        chex.assert_shape(cost, ())
        penalized = proposed_loss[0] - lambda_ * cost
        chex.assert_shape(penalized, ())
        temper = (robust_config.objective in ("entropic", "relaxed_kl_cvar")
                  or robust_config.ula_temper)
        score = penalized / temperature if temper else penalized
        chex.assert_shape(score, ())
        return score

    proposals, log_weights = sample_langevin_proposal(
        proposal_key, anchors, log_target, robust_config)
    n_proposals = proposals.shape[1]
    chex.assert_shape(proposals, (batch, n_proposals, dimension))
    chex.assert_shape(log_weights, (batch, n_proposals))
    flat_proposals = proposals.reshape(batch * n_proposals, dimension)
    chex.assert_shape(flat_proposals, (batch * n_proposals, dimension))
    physical_proposals = (flat_proposals * robust_state.scale[None, :]
                          + robust_state.location[None, :])
    chex.assert_shape(physical_proposals, (batch * n_proposals, dimension))
    proposed_prices, proposed_features, _ = decode_trajectory(physical_proposals, market)
    chex.assert_shape(proposed_prices, (batch * n_proposals, n_time + 1, expected_assets))
    chex.assert_shape(proposed_features, (batch * n_proposals, n_time, n_features))
    flat_losses, flat_diagnostics, updated = pathwise_hedging_loss(
        strategy, proposed_prices, proposed_features, epsilon, strategy_config, training,
        market.strike)
    chex.assert_shape(flat_losses, (batch * n_proposals,))
    pathwise_losses = flat_losses.reshape(batch, n_proposals)
    chex.assert_shape(pathwise_losses, (batch, n_proposals))
    if robust_config.enforce_convex_order:
        order_lambda_ij = jax.nn.softplus(robust_state.raw_lambda_ij)
        flat_order_penalty = convex_order_penalty(
            proposed_prices[..., 0], jnp.asarray(robust_config.convex_order_strikes,
                                                  dtype=proposed_prices.dtype), order_lambda_ij)
        chex.assert_shape(flat_order_penalty, (batch * n_proposals,))
        pathwise_losses = pathwise_losses + flat_order_penalty.reshape(batch, n_proposals)
        chex.assert_shape(pathwise_losses, (batch, n_proposals))
    transport_cost = quadratic_transport_cost(
        proposals, anchors, reference_scale, robust_config.cost_power)
    chex.assert_shape(transport_cost, (batch, n_proposals))
    if robust_config.objective == "entropic":
        objective = entropic_dual(
            robust_state, pathwise_losses, transport_cost, log_weights, robust_config)
    elif robust_config.objective == "per_sample_cvar":
        objective = per_sample_cvar_dual(
            robust_state, pathwise_losses, transport_cost, log_weights, robust_config)
    elif robust_config.objective == "relaxed_kl_cvar":
        objective = relaxed_kl_cvar_dual(
            robust_state, pathwise_losses, transport_cost, log_weights, robust_config)
    elif robust_config.objective == "plain_average":
        objective = plain_average_dual(
            robust_state, pathwise_losses, transport_cost, log_weights, robust_config)
    elif robust_config.objective == "plain_cvar_of_average":
        objective = plain_cvar_of_average_dual(
            robust_state, pathwise_losses, transport_cost, log_weights, robust_config)
    else:
        objective = joint_cvar_dual(
            robust_state, pathwise_losses, transport_cost, log_weights, robust_config)
    chex.assert_shape(objective, ())
    return objective, (pathwise_losses, transport_cost, log_weights, *flat_diagnostics), updated


def merge_batchnorm_state(strategy: Strategy, bn_strategy: Strategy) -> Strategy:
    """Fold BN running stats from a forward pass into a strategy; inputs old/BN-updated strategy,
    output the merged strategy. GRUCore has no train/eval-dependent state, so this is a no-op then."""

    if isinstance(strategy.core, MLPCore):
        new_core = eqx.tree_at(lambda c: (c.mean1, c.var1, c.mean2, c.var2), strategy.core,
                               (bn_strategy.core.mean1, bn_strategy.core.var1,
                                bn_strategy.core.mean2, bn_strategy.core.var2))
        return eqx.tree_at(lambda s: s.core, strategy, new_core)
    return strategy


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
        strategy = merge_batchnorm_state(strategy, bn_strategy)
        grad_norm = optax.global_norm(grads)
        chex.assert_shape(loss, ())
        chex.assert_shape(grad_norm, ())
        return strategy, opt_state, loss, grad_norm

    @eqx.filter_jit
    def evaluate(strategy: Strategy, prices: Array, features: Array):
        """Evaluate frozen-BN policy; inputs strategy/paths, outputs loss and pathwise diagnostics."""

        loss, diagnostics, _ = hedging_loss(strategy, prices, features, epsilon, strategy_config, False, alpha, strike)
        return loss, diagnostics

    return train_step, evaluate


def make_robust_steps(
    optimizer: optax.GradientTransformation,
    market: MarketConfig,
    strategy_config: StrategyConfig,
    robust_config: RobustConfig,
    epsilon: float,
):
    """Create top-level WDRO train/evaluation callables; inputs are optimizer/configs, outputs compiled steps."""

    @eqx.filter_jit
    def train_step(
        strategy: Strategy,
        robust_state: RobustState,
        opt_state: optax.OptState,
        prices: Array,
        variance: Array | None,
        proposal_key: Array,
    ):
        """Run one compiled WDRO Adam step; inputs are states/base paths/key, outputs updated states/loss/norm."""

        def objective(parameters: tuple[Strategy, RobustState]):
            """Differentiate the WDRO objective; input is model/dual state, output is loss and BN strategy."""

            candidate, candidate_robust = parameters
            loss, _, updated = robust_hedging_loss(
                candidate, candidate_robust, prices, variance, proposal_key, market,
                strategy_config, robust_config, epsilon, True)
            return loss, updated

        parameters = (strategy, robust_state)
        (loss, bn_strategy), grads = eqx.filter_value_and_grad(objective, has_aux=True)(parameters)
        strategy_grads, robust_grads = grads
        strategy_grads = eqx.tree_at(
            lambda candidate: (candidate.cash, candidate.q), strategy_grads,
            (jnp.zeros_like(strategy_grads.cash), jnp.zeros_like(strategy_grads.q)))
        if (robust_config.objective in ("entropic", "per_sample_cvar")
                or (robust_config.objective == "joint_cvar" and robust_config.exact_eta)):
            robust_grads = eqx.tree_at(
                lambda candidate: candidate.eta, robust_grads,
                jnp.zeros_like(robust_grads.eta))
        if robust_config.fix_lambda:
            robust_grads = eqx.tree_at(
                lambda candidate: candidate.raw_lambda, robust_grads,
                jnp.zeros_like(robust_grads.raw_lambda))
        if not robust_config.enforce_convex_order or robust_config.fix_convex_order_lambda:
            robust_grads = eqx.tree_at(
                lambda candidate: candidate.raw_lambda_ij, robust_grads,
                jnp.zeros_like(robust_grads.raw_lambda_ij))
        robust_grads = eqx.tree_at(
            lambda candidate: (candidate.location, candidate.scale), robust_grads,
            (jnp.zeros_like(robust_grads.location), jnp.zeros_like(robust_grads.scale)))
        grads = (strategy_grads, robust_grads)
        updates, opt_state = optimizer.update(grads, opt_state, parameters)
        strategy, robust_state = eqx.apply_updates(parameters, updates)
        strategy = merge_batchnorm_state(strategy, bn_strategy)
        grad_norm = optax.global_norm(grads)
        chex.assert_shape(loss, ())
        chex.assert_shape(grad_norm, ())
        return strategy, robust_state, opt_state, loss, grad_norm

    @eqx.filter_jit
    def evaluate(
        strategy: Strategy,
        robust_state: RobustState,
        prices: Array,
        variance: Array | None,
        proposal_key: Array,
    ):
        """Evaluate a frozen-BN WDRO policy; inputs are states/base paths/key, outputs objective/diagnostics."""

        loss, diagnostics, _ = robust_hedging_loss(
            strategy, robust_state, prices, variance, proposal_key, market,
            strategy_config, robust_config, epsilon, False)
        return loss, diagnostics

    return train_step, evaluate


def train(
    config: Config,
    train_dataset: tuple[Array, Array, Array | None] | None = None,
    validation_dataset: tuple[Array, Array, Array | None] | None = None,
) -> tuple[Strategy, RobustState | None, dict[str, list[float]]]:
    """Train nominal or WDRO hedging with fixed validation; input config, outputs best model/dual/history.

    If `train_dataset`/`validation_dataset` are given (each a `(prices, features, variance)`
    triple, `variance` possibly `None`), training samples minibatches with replacement from the
    fixed `train_dataset` pool instead of simulating fresh iid paths every step, and validation
    uses `validation_dataset` as-is. This is what a finite training-sample-size `n` study needs;
    passing neither reproduces the original iid-per-step behavior exactly.
    """

    if config.robust.enabled:
        if config.robust.rho < 0.0 or config.robust.kl_epsilon <= 0.0 or config.robust.tau < 0.0:
            raise ValueError("rho and tau must be nonnegative and kl_epsilon must be positive")
        if not 0.0 < config.robust.alpha < 1.0:
            raise ValueError("robust alpha must lie strictly between zero and one")
        if config.robust.proposal_samples < 1 or config.robust.initial_lambda <= 0.0:
            raise ValueError("proposal_samples and initial_lambda must be positive")
    key = jax.random.PRNGKey(config.training.seed)
    key, init_key, standardization_key, validation_key, validation_proposal_key = jax.random.split(key, 5)
    strategy = initialize_strategy(init_key, config.market, config.strategy)
    if config.training.grad_clip_norm is None:
        optimizer = optax.adam(config.training.learning_rate)
    else:
        optimizer = optax.chain(optax.clip_by_global_norm(config.training.grad_clip_norm),
                                optax.adam(config.training.learning_rate))
    simulate_training = jax.jit(lambda simulation_key: simulate_market(
        simulation_key, config.market, config.training.batch_size))
    simulate_validation = jax.jit(lambda simulation_key: simulate_market(
        simulation_key, config.market, config.training.validation_size))
    if validation_dataset is not None:
        validation_prices, validation_features, validation_variance = validation_dataset
    else:
        validation_prices, validation_features, validation_variance = simulate_validation(validation_key)
    train_prices = train_features = train_variance = None
    n_pool = 0
    if train_dataset is not None:
        train_prices, train_features, train_variance = train_dataset
        n_pool = train_prices.shape[0]
    raw_lambda = jnp.log(jnp.expm1(jnp.asarray(config.robust.initial_lambda, dtype=strategy.cash.dtype)))
    chex.assert_shape(raw_lambda, ())
    eta = jnp.asarray(config.robust.initial_eta, dtype=strategy.cash.dtype)
    chex.assert_shape(eta, ())
    if config.robust.enabled:
        if train_dataset is not None:
            # Standardize from the training pool itself (not validation) to avoid leaking
            # held-out-fold statistics into the anchor location/scale.
            standardization_prices, standardization_variance = train_prices, train_variance
        else:
            standardization_prices, _, standardization_variance = simulate_validation(
                standardization_key)
        standardization_coordinates = encode_trajectory(
            standardization_prices, standardization_variance, config.market)
        n_standardization = standardization_coordinates.shape[0]
        dimension = standardization_coordinates.shape[1]
        chex.assert_shape(
            standardization_coordinates, (n_standardization, dimension))
        coordinate_location = jnp.mean(standardization_coordinates, axis=0)
        chex.assert_shape(coordinate_location, (dimension,))
        coordinate_scale = jnp.maximum(jnp.std(standardization_coordinates, axis=0), 1e-3)
        chex.assert_shape(coordinate_scale, (dimension,))
        n_strikes = len(config.robust.convex_order_strikes)
        raw_lambda_ij_init = jnp.log(jnp.expm1(jnp.asarray(
            config.robust.convex_order_lambda_init, dtype=strategy.cash.dtype)))
        raw_lambda_ij = jnp.full((n_strikes, config.market.n_time), raw_lambda_ij_init,
                                 dtype=strategy.cash.dtype)
        chex.assert_shape(raw_lambda_ij, (n_strikes, config.market.n_time))
        robust_state = RobustState(raw_lambda, eta, coordinate_location, coordinate_scale, raw_lambda_ij)
    else:
        robust_state = None
    parameters = (strategy, robust_state) if robust_state is not None else strategy
    opt_state = optimizer.init(eqx.filter(parameters, eqx.is_array))
    if robust_state is not None:
        train_step, evaluate = make_robust_steps(
            optimizer, config.market, config.strategy, config.robust, config.epsilon)
    else:
        train_step, evaluate = make_steps(
            optimizer, config.strategy, config.epsilon, config.alpha, config.market.strike)
    calibration_config = dataclasses.replace(config.strategy, bn_momentum=0.0)

    @eqx.filter_jit
    def calibrate(candidate: Strategy) -> Strategy:
        """Recompute exact fixed-validation BN statistics; input strategy, output calibrated strategy."""

        _, calibrated = strategy_apply(candidate, validation_features, calibration_config, True)
        return calibrated

    best_strategy = strategy
    best_robust_state = robust_state
    best_loss = float("inf")
    history: dict[str, list[float]] = {
        "step": [], "train_loss": [], "validation_loss": [], "grad_norm": [],
        "lambda": [], "eta": []}
    for step in range(config.training.steps):
        key, batch_key, proposal_key = jax.random.split(key, 3)
        if train_dataset is not None:
            indices = jax.random.randint(batch_key, (config.training.batch_size,), 0, n_pool)
            prices = train_prices[indices]
            features = train_features[indices]
            variance = None if train_variance is None else train_variance[indices]
        else:
            prices, features, variance = simulate_training(batch_key)
        if robust_state is None:
            strategy, opt_state, loss, grad_norm = train_step(
                strategy, opt_state, prices, features)
        else:
            strategy, robust_state, opt_state, loss, grad_norm = train_step(
                strategy, robust_state, opt_state, prices, variance, proposal_key)
        if step % config.training.validation_frequency == 0 or step == config.training.steps - 1:
            strategy = calibrate(strategy)
            if robust_state is None:
                validation_loss, _ = evaluate(strategy, validation_prices, validation_features)
                lambda_value = eta_value = 0.0
            else:
                validation_loss, _ = evaluate(
                    strategy, robust_state, validation_prices, validation_variance,
                    validation_proposal_key)
                lambda_value = float(jax.nn.softplus(robust_state.raw_lambda))
                eta_value = float(robust_state.eta)
            value = float(validation_loss)
            history["step"].append(step)
            history["train_loss"].append(float(loss))
            history["validation_loss"].append(value)
            history["grad_norm"].append(float(grad_norm))
            history["lambda"].append(lambda_value)
            history["eta"].append(eta_value)
            if value < best_loss:
                best_loss = value
                best_strategy = jax.tree.map(lambda x: jnp.array(x), strategy)
                best_robust_state = None if robust_state is None else jax.tree.map(
                    lambda x: jnp.array(x), robust_state)
            print(f"step={step:6d} train={float(loss):.6f} validation={value:.6f} "
                  f"lambda={lambda_value:.4f} eta={eta_value:.4f}")
    return best_strategy, best_robust_state, history


def evaluate_chunks(
    strategy: Strategy, robust_state: RobustState | None, config: Config,
) -> dict[str, float | list[float]]:
    """Evaluate independent chunks; inputs trained model/dual/config, output aggregate metrics dictionary."""

    paths = 1_000_000 if config.evaluation.paper_scale else config.evaluation.paths
    optimizer = optax.adam(config.training.learning_rate)
    _, evaluate = make_steps(optimizer, config.strategy, config.epsilon,
                             config.alpha, config.market.strike)
    robust_evaluate = None
    if robust_state is not None:
        _, robust_evaluate = make_robust_steps(
            optimizer, config.market, config.strategy, config.robust, config.epsilon)
    chunk_limit = config.evaluation.chunk_size
    if robust_state is not None:
        chunk_limit = max(1, chunk_limit // config.robust.proposal_samples)
    simulate_full_chunk = jax.jit(lambda simulation_key: simulate_market(
        simulation_key, config.market, chunk_limit))
    key = jax.random.PRNGKey(config.training.seed + 10_000)
    errors, costs, turnovers, holdings, turnover_dates = [], [], [], [], []
    chunk_sizes, robust_values = [], []
    robust_transport_means, robust_loss_means, robust_ess_means = [], [], []
    remaining = paths
    while remaining:
        size = min(remaining, chunk_limit)
        key, chunk_key, proposal_key = jax.random.split(key, 3)
        if size == chunk_limit:
            prices, features, variance = simulate_full_chunk(chunk_key)
        else:
            simulate_remainder = jax.jit(lambda simulation_key: simulate_market(
                simulation_key, config.market, size))
            prices, features, variance = simulate_remainder(chunk_key)
        _, diagnostics = evaluate(strategy, prices, features)
        error, _, cost, turnover, chunk_holdings = diagnostics
        if robust_state is not None and robust_evaluate is not None:
            robust_value, robust_diagnostics = robust_evaluate(
                strategy, robust_state, prices, variance, proposal_key)
            robust_losses, robust_transport, robust_log_weights = robust_diagnostics[:3]
            n_proposals = robust_losses.shape[1]
            chex.assert_shape(robust_losses, (size, n_proposals))
            chex.assert_shape(robust_transport, (size, n_proposals))
            chex.assert_shape(robust_log_weights, (size, n_proposals))
            log_weights_np = np.asarray(robust_log_weights)
            shifted_weights = np.exp(
                log_weights_np - np.max(log_weights_np, axis=1, keepdims=True))
            effective_sample_size = np.square(np.sum(shifted_weights, axis=1)) / np.sum(
                np.square(shifted_weights), axis=1)
            robust_values.append(float(robust_value))
            robust_transport_means.append(float(np.mean(np.asarray(robust_transport))))
            robust_loss_means.append(float(np.mean(np.asarray(robust_losses))))
            robust_ess_means.append(float(np.mean(effective_sample_size)))
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
    metrics = {"mse": float(np.mean(error_np**2)), "cvar": cvar, "cash": float(strategy.cash),
               "error_mean": float(np.mean(error_np)), "error_std": float(np.std(error_np)),
               "error_quantiles": np.quantile(error_np, [0.01, 0.05, 0.5, 0.95, 0.99]).tolist(),
               "mean_cost": float(np.mean(cost_np)), "mean_turnover": float(np.mean(turnover_np)),
               "mean_holdings": np.average(np.stack(holdings), axis=0, weights=chunk_sizes).tolist(),
               "mean_turnover_by_date": np.average(np.stack(turnover_dates), axis=0,
                                                    weights=chunk_sizes).tolist(),
               "errors": error_np.tolist()}
    if robust_state is not None:
        metrics["robust_objective"] = float(np.average(robust_values, weights=chunk_sizes))
        metrics["lambda"] = float(jax.nn.softplus(robust_state.raw_lambda))
        metrics["eta"] = float(robust_state.eta)
        metrics["mean_transport_cost"] = float(np.average(
            robust_transport_means, weights=chunk_sizes))
        metrics["mean_proposed_loss"] = float(np.average(
            robust_loss_means, weights=chunk_sizes))
        metrics["mean_importance_ess"] = float(np.average(
            robust_ess_means, weights=chunk_sizes))
        metrics["proposal_samples"] = config.robust.proposal_samples
        metrics["proposal_scale"] = config.robust.proposal_scale
        metrics["ula_steps"] = config.robust.ula_steps
    return metrics


def save_artifacts(
    strategy: Strategy,
    history: dict[str, list[float]],
    metrics: dict,
    config: Config,
    robust_state: RobustState | None = None,
) -> Path:
    """Save config/model/metrics/plots; inputs completed run objects, output artifact directory path."""

    suffix = f"{config.market.name}_cost_{config.epsilon:g}"
    if robust_state is not None:
        suffix += f"_{config.robust.objective}_ula_{config.robust.ula_steps}"
    directory = Path(config.evaluation.output_dir) / suffix
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2))
    diagnostic_prices = diagnostic_learned = diagnostic_analytic = None
    if config.market.name == "black_scholes" and config.epsilon == 0.0:
        diagnostic_prices, diagnostic_features = simulate_black_scholes(
            jax.random.PRNGKey(config.training.seed + 20_000), config.market, 4_096)
        diagnostic_learned, _ = strategy_apply(strategy, diagnostic_features, config.strategy, False)
        times = jnp.arange(config.market.n_time, dtype=strategy.cash.dtype) * (config.market.horizon / config.market.n_time)
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
        delta_correlation_value = float(delta_correlation)
        metrics["analytic_delta_correlation"] = (
            delta_correlation_value if np.isfinite(delta_correlation_value) else None)
    serializable_metrics = {key: value for key, value in metrics.items() if key != "errors"}
    (directory / "metrics.json").write_text(json.dumps(serializable_metrics, indent=2))
    (directory / "history.json").write_text(json.dumps(history, indent=2))
    eqx.tree_serialise_leaves(directory / "model.eqx", strategy)
    if robust_state is not None:
        eqx.tree_serialise_leaves(directory / "robust.eqx", robust_state)
    figure, axis = plt.subplots()
    axis.plot(history["step"], history["train_loss"], label="train")
    axis.plot(history["step"], history["validation_loss"], label="validation")
    all_losses = np.asarray(history["train_loss"] + history["validation_loss"])
    axis.set_yscale("log" if np.all(all_losses > 0.0) else "symlog")
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
    comparison_name = "comparison.csv" if robust_state is None else "robust_comparison.csv"
    comparison = Path(config.evaluation.output_dir) / comparison_name
    rows = {}
    if comparison.exists():
        for line in comparison.read_text().splitlines()[1:]:
            fields = line.split(",")
            if robust_state is None and len(fields) == 6:
                fields = [fields[0], fields[1], "0", "float32", "100000", *fields[2:]]
            row_key = (fields[0], fields[1]) if robust_state is None else (
                fields[0], fields[1], fields[2], fields[3])
            rows[row_key] = fields
    evaluation_paths = 1_000_000 if config.evaluation.paper_scale else config.evaluation.paths
    if robust_state is None:
        row = [config.market.name, f"{config.epsilon:g}", str(config.training.seed), config.market.dtype,
               str(evaluation_paths), f"{config.alpha:g}", str(serializable_metrics["mse"]),
               str(serializable_metrics["cvar"]), str(serializable_metrics["cash"]),
               str(serializable_metrics["mean_cost"]), str(serializable_metrics["mean_turnover"])]
        rows[(row[0], row[1])] = row
        header = "market,epsilon,seed,dtype,evaluation_paths,alpha,mse,cvar,cash,mean_cost,mean_turnover\n"
    else:
        row = [config.market.name, f"{config.epsilon:g}", config.robust.objective,
               str(config.robust.ula_steps), str(config.training.seed), config.market.dtype,
               str(evaluation_paths), f"{config.robust.alpha:g}",
               str(serializable_metrics["robust_objective"]), str(serializable_metrics["lambda"]),
               str(serializable_metrics["eta"]), str(serializable_metrics["cash"]),
               str(serializable_metrics["mean_cost"]), str(serializable_metrics["mean_turnover"])]
        rows[(row[0], row[1], row[2], row[3])] = row
        header = ("market,epsilon,objective,ula_steps,seed,dtype,evaluation_paths,alpha,"
                  "robust_objective,lambda,eta,cash,mean_cost,mean_turnover\n")
    comparison.write_text(header + "\n".join(",".join(value) for value in rows.values()) + "\n")
    return directory


def main(config: Config) -> None:
    """Run a CLI experiment; input is Tyro Config and output consists of printed/saved artifacts."""

    if config.market.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    strategy, robust_state, history = train(config)
    metrics = evaluate_chunks(strategy, robust_state, config)
    if robust_state is None:
        metrics["final_train_cvar"] = history["train_loss"][-1]
        metrics["best_validation_cvar"] = min(history["validation_loss"])
        indifference_price = metrics["cvar"]
    else:
        metrics["final_train_robust"] = history["train_loss"][-1]
        metrics["best_validation_robust"] = min(history["validation_loss"])
        indifference_price = metrics["robust_objective"]
    strategy = eqx.tree_at(
        lambda candidate: candidate.cash, strategy,
        jnp.asarray(indifference_price, dtype=strategy.cash.dtype))
    metrics["cash"] = indifference_price
    directory = save_artifacts(strategy, history, metrics, config, robust_state)
    print(json.dumps({key: value for key, value in metrics.items() if key != "errors"}, indent=2))
    print(f"saved={directory}")


if __name__ == "__main__":
    main(tyro.cli(Config))

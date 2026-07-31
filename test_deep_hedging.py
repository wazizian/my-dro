"""Pytest coverage for simulation, strategy, accounting, optimization, and artifacts."""

import dataclasses

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from deep_hedging import (
    Config,
    MarketConfig,
    StrategyConfig,
    decode_trajectory,
    encode_trajectory,
    hedging_loss,
    initialize_strategy,
    make_robust_steps,
    make_steps,
    market_dimensions,
    save_artifacts,
    simulate_black_scholes,
    simulate_heston,
    strategy_apply,
)
from wdro import RobustConfig, RobustState


def test_black_scholes_shape_positive_and_deterministic():
    """Verify exact BS simulation shape, positivity, finiteness, and fixed-key determinism."""

    config = MarketConfig(n_time=5, horizon=5 / 365)
    key = jax.random.PRNGKey(1)
    prices, features = simulate_black_scholes(key, config, 16)
    chex.assert_shape(prices, (16, 6, 1))
    chex.assert_shape(features, (16, 5, 1))
    assert bool(jnp.all(jnp.isfinite(prices)))
    assert bool(jnp.all(prices > 0))
    repeated, _ = simulate_black_scholes(key, config, 16)
    np.testing.assert_array_equal(prices, repeated)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_heston_shapes_finiteness_and_terminal_swap(dtype):
    """Verify Heston paths, including nonnegative CIR variance and terminal accumulated variance."""

    if dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    config = MarketConfig(name="heston", n_time=5, horizon=5 / 365, dtype=dtype)
    prices, features, variance = simulate_heston(jax.random.PRNGKey(2), config, 32)
    chex.assert_shape(prices, (32, 6, 2))
    chex.assert_shape(features, (32, 5, 2))
    chex.assert_shape(variance, (32, 6))
    assert bool(jnp.all(jnp.isfinite(prices)))
    assert bool(jnp.all(jnp.isfinite(variance)))
    assert bool(jnp.all(prices[:, :, 0] > 0))
    assert bool(jnp.all(variance >= 0))
    # The terminal adjustment L(T,V_T) is exactly zero.
    dt = config.horizon / config.n_time
    accumulated = jnp.sum(variance[:, :-1] * dt, axis=1)
    np.testing.assert_allclose(prices[:, -1, 1], accumulated, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("name", ["black_scholes", "heston"])
def test_market_trajectory_coordinates_round_trip(name):
    """Verify both markets encode and decode into the same valid prices, features, and variance."""

    market = MarketConfig(name=name, n_time=3, horizon=3 / 365)
    prices, features, variance = (
        (*simulate_black_scholes(jax.random.PRNGKey(22), market, 8), None)
        if name == "black_scholes"
        else simulate_heston(jax.random.PRNGKey(22), market, 8)
    )
    coordinates = encode_trajectory(prices, variance, market)
    expected_dimension = market.n_time if name == "black_scholes" else 2 * market.n_time
    chex.assert_shape(coordinates, (8, expected_dimension))
    decoded_prices, decoded_features, decoded_variance = decode_trajectory(coordinates, market)
    np.testing.assert_allclose(decoded_prices, prices, rtol=3e-5, atol=1e-7)
    np.testing.assert_allclose(decoded_features, features, rtol=3e-5, atol=1e-7)
    assert bool(jnp.all(decoded_prices[:, :, 0] > 0.0))
    if variance is not None:
        np.testing.assert_allclose(decoded_variance, variance, rtol=3e-5, atol=1e-7)
        assert bool(jnp.all(decoded_variance > 0.0))


def test_strategy_widths_output_and_batch_norm_modes():
    """Verify both paper architectures and that BN updates only in training mode."""

    for name, expected in (("black_scholes", (1, 1)), ("heston", (2, 2))):
        market = MarketConfig(name=name, n_time=3)
        n_assets, n_features = market_dimensions(market)
        assert (n_assets, n_features) == expected
        strategy = initialize_strategy(jax.random.PRNGKey(n_assets), market, StrategyConfig())
        features = jax.random.normal(jax.random.PRNGKey(8), (8, 3, n_features))
        holdings, trained = strategy_apply(strategy, features, StrategyConfig(), True)
        chex.assert_shape(holdings, (8, 3, n_assets))
        assert not np.array_equal(strategy.core.mean1, trained.core.mean1)
        _, inferred = strategy_apply(trained, features, StrategyConfig(), False)
        np.testing.assert_array_equal(trained.core.mean1, inferred.core.mean1)


def test_previous_holding_affects_later_decision():
    """Verify the recurrent previous-holding input can alter the following date's network output."""

    market = MarketConfig(n_time=2)
    strategy = initialize_strategy(jax.random.PRNGKey(18), market, StrategyConfig())
    features = jnp.full((4, 2, 1), jnp.log(100.0))
    baseline, _ = strategy_apply(strategy, features, StrategyConfig(), False)
    changed_bias = strategy.core.b3.at[0, 0].add(1.0)
    changed = eqx.tree_at(lambda candidate: candidate.core.b3, strategy, changed_bias)
    perturbed, _ = strategy_apply(changed, features, StrategyConfig(), False)
    assert not np.array_equal(baseline[:, 1], perturbed[:, 1])


def test_zero_and_constant_strategy_accounting():
    """Verify zero gains/costs and that constant holdings incur only the initial trade cost."""

    market = MarketConfig(n_time=3, horizon=3 / 365)
    strategy = initialize_strategy(jax.random.PRNGKey(3), market, StrategyConfig())
    strategy = eqx.tree_at(lambda s: (s.core.w1, s.core.b1, s.core.w2, s.core.b2, s.core.w3, s.core.b3), strategy,
                           tuple(jnp.zeros_like(x) for x in (strategy.core.w1, strategy.core.b1, strategy.core.w2,
                                                            strategy.core.b2, strategy.core.w3, strategy.core.b3)))
    prices = jnp.asarray([[[100.0], [101.0], [99.0], [102.0]]])
    features = jnp.log(prices[:, :-1])
    _, (_, gains, costs, _, holdings), _ = hedging_loss(strategy, prices, features, 0.01, StrategyConfig(), False)
    np.testing.assert_array_equal(holdings, 0.0)
    np.testing.assert_array_equal(gains, 0.0)
    np.testing.assert_array_equal(costs, 0.0)
    strategy = eqx.tree_at(lambda s: s.core.b3, strategy, jnp.ones_like(strategy.core.b3))
    _, (_, gains, costs, _, holdings), _ = hedging_loss(strategy, prices, features, 0.01, StrategyConfig(), False)
    np.testing.assert_allclose(holdings, 1.0)
    np.testing.assert_allclose(gains, 2.0)
    np.testing.assert_allclose(costs, 1.0)


def test_costs_are_exactly_zero_or_nonnegative():
    """Verify epsilon zero removes costs exactly and positive epsilon produces nonnegative costs."""

    market = MarketConfig(n_time=3, horizon=3 / 365)
    strategy = initialize_strategy(jax.random.PRNGKey(13), market, StrategyConfig())
    prices, features = simulate_black_scholes(jax.random.PRNGKey(14), market, 16)
    _, (_, _, zero_cost, _, _), _ = hedging_loss(strategy, prices, features, 0.0, StrategyConfig(), False)
    _, (_, _, positive_cost, _, _), _ = hedging_loss(strategy, prices, features, 0.01, StrategyConfig(), False)
    np.testing.assert_array_equal(zero_cost, 0.0)
    assert bool(jnp.all(positive_cost >= 0.0))
    assert bool(jnp.any(positive_cost > 0.0))


def test_hedging_loss_matches_cvar_definition():
    """Verify CVaR equals q plus the scaled mean excess loss for a known loss sample."""

    market = MarketConfig(n_time=1, horizon=1 / 365)
    strategy = initialize_strategy(jax.random.PRNGKey(19), market, StrategyConfig())
    network_leaves = (strategy.core.w1, strategy.core.b1, strategy.core.w2,
                      strategy.core.b2, strategy.core.w3, strategy.core.b3)
    strategy = eqx.tree_at(
        lambda candidate: (candidate.core.w1, candidate.core.b1, candidate.core.w2,
                           candidate.core.b2, candidate.core.w3, candidate.core.b3),
        strategy,
        tuple(jnp.zeros_like(leaf) for leaf in network_leaves),
    )
    strategy = eqx.tree_at(lambda candidate: candidate.q, strategy, jnp.asarray(1.0))
    terminal = jnp.asarray([100.0, 101.0, 102.0, 103.0])
    chex.assert_shape(terminal, (4,))
    initial = jnp.full((4,), 100.0)
    chex.assert_shape(initial, (4,))
    prices = jnp.stack((initial, terminal), axis=1)[:, :, None]
    chex.assert_shape(prices, (4, 2, 1))
    features = jnp.log(prices[:, :-1])
    chex.assert_shape(features, (4, 1, 1))
    loss, _, _ = hedging_loss(
        strategy, prices, features, 0.0, StrategyConfig(), False, alpha=0.5)
    chex.assert_shape(loss, ())
    np.testing.assert_allclose(loss, 2.5)


def test_jitted_optimizer_step_is_finite_and_updates_parameters():
    """Verify one top-level compiled Adam step returns finite values and changes trainable leaves."""

    market = MarketConfig(n_time=3, horizon=3 / 365)
    strategy = initialize_strategy(jax.random.PRNGKey(4), market, StrategyConfig())
    prices, features = simulate_black_scholes(jax.random.PRNGKey(5), market, 32)
    optimizer = optax.adam(0.005)
    state = optimizer.init(eqx.filter(strategy, eqx.is_array))
    step, _ = make_steps(optimizer, StrategyConfig(), 0.0)
    updated, _, loss, norm = step(strategy, state, prices, features)
    assert bool(jnp.isfinite(loss))
    assert bool(jnp.isfinite(norm))
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(strategy), jax.tree.leaves(updated)))


@pytest.mark.parametrize("name", ["black_scholes", "heston"])
@pytest.mark.parametrize("objective", ["entropic", "joint_cvar"])
def test_jitted_robust_optimizer_step_is_finite(name, objective):
    """Verify both WDRO objectives compile and update model/dual state for both supported markets."""

    market = MarketConfig(name=name, n_time=2, horizon=2 / 365)
    strategy = initialize_strategy(jax.random.PRNGKey(23), market, StrategyConfig())
    dimension = market.n_time if name == "black_scholes" else 2 * market.n_time
    robust_state = RobustState(
        jnp.asarray(0.0), jnp.asarray(0.0), jnp.zeros((dimension,)), jnp.ones((dimension,)))
    robust_config = RobustConfig(enabled=True, objective=objective, proposal_samples=2,
                                 proposal_scale=0.01, ula_steps=0)
    prices, _, variance = (
        (*simulate_black_scholes(jax.random.PRNGKey(24), market, 4), None)
        if name == "black_scholes"
        else simulate_heston(jax.random.PRNGKey(24), market, 4)
    )
    optimizer = optax.adam(0.001)
    parameters = (strategy, robust_state)
    state = optimizer.init(eqx.filter(parameters, eqx.is_array))
    step, _ = make_robust_steps(optimizer, market, StrategyConfig(), robust_config, 0.0)
    updated_strategy, updated_robust, _, loss, norm = step(
        strategy, robust_state, state, prices, variance, jax.random.PRNGKey(25))
    chex.assert_shape(loss, ())
    chex.assert_shape(norm, ())
    chex.assert_shape(updated_robust.raw_lambda, ())
    chex.assert_shape(updated_robust.eta, ())
    assert bool(jnp.isfinite(loss))
    assert bool(jnp.isfinite(norm))
    np.testing.assert_array_equal(updated_strategy.cash, strategy.cash)
    np.testing.assert_array_equal(updated_strategy.q, strategy.q)
    np.testing.assert_array_equal(updated_robust.location, robust_state.location)
    np.testing.assert_array_equal(updated_robust.scale, robust_state.scale)
    if objective == "entropic":
        np.testing.assert_array_equal(updated_robust.eta, robust_state.eta)
    assert any(not np.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(strategy), jax.tree.leaves(updated_strategy)))


def test_short_fixed_batch_training_reduces_loss():
    """Verify fixed validation CVaR falls after training on a separate fixed batch."""

    market = MarketConfig(n_time=3, horizon=3 / 365)
    strategy = initialize_strategy(jax.random.PRNGKey(6), market, StrategyConfig())
    prices, features = simulate_black_scholes(jax.random.PRNGKey(7), market, 128)
    validation_prices, validation_features = simulate_black_scholes(jax.random.PRNGKey(17), market, 128)
    optimizer = optax.adam(0.005)
    state = optimizer.init(eqx.filter(strategy, eqx.is_array))
    strategy_config = StrategyConfig()
    step, evaluate = make_steps(optimizer, strategy_config, 0.0)
    calibration = dataclasses.replace(strategy_config, bn_momentum=0.0)
    _, initial_strategy = strategy_apply(strategy, validation_features, calibration, True)
    initial, _ = evaluate(initial_strategy, validation_prices, validation_features)
    for _ in range(30):
        strategy, state, _, _ = step(strategy, state, prices, features)
    _, strategy = strategy_apply(strategy, validation_features, calibration, True)
    final, _ = evaluate(strategy, validation_prices, validation_features)
    assert float(final) < float(initial)


def test_artifacts_and_checkpoint_round_trip(tmp_path):
    """Verify required artifacts are saved and an Equinox checkpoint restores identical leaves."""

    config = Config()
    config.evaluation.output_dir = str(tmp_path)
    config.market.n_time = 2
    strategy = initialize_strategy(jax.random.PRNGKey(15), config.market, config.strategy)
    history = {"step": [0], "train_loss": [1.0], "validation_loss": [1.1], "grad_norm": [0.5]}
    metrics = {"mse": 1.2, "cvar": 1.5, "cash": 0.0, "error_mean": 0.0, "error_std": 1.0,
               "error_quantiles": [0.0] * 5, "mean_cost": 0.0, "mean_turnover": 0.0,
               "mean_holdings": [[0.0], [0.0]], "mean_turnover_by_date": [0.0, 0.0],
               "errors": [0.0, 1.0]}
    directory = save_artifacts(strategy, history, metrics, config)
    expected = {"config.json", "metrics.json", "history.json", "model.eqx", "loss.png",
                "hedging_error.png", "holdings.png", "turnover.png", "analytic_delta.png"}
    assert expected.issubset({path.name for path in directory.iterdir()})
    restored = eqx.tree_deserialise_leaves(directory / "model.eqx", strategy)
    for original, loaded in zip(jax.tree.leaves(strategy), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(original, loaded)


def test_robust_artifacts_and_dual_checkpoint_round_trip(tmp_path):
    """Verify robust runs save their scalar dual variables and fixed coordinate statistics."""

    market = MarketConfig(name="heston", n_time=2, horizon=2 / 365)
    robust_config = RobustConfig(enabled=True, objective="joint_cvar", ula_steps=1)
    config = Config(market=market, robust=robust_config)
    config.evaluation.output_dir = str(tmp_path)
    strategy = initialize_strategy(jax.random.PRNGKey(26), market, config.strategy)
    robust_state = RobustState(
        jnp.asarray(0.2), jnp.asarray(0.3), jnp.zeros((4,)), jnp.ones((4,)))
    history = {"step": [0], "train_loss": [1.0], "validation_loss": [1.1],
               "grad_norm": [0.5], "lambda": [0.8], "eta": [0.3]}
    metrics = {"mse": 1.2, "cvar": 1.5, "robust_objective": 1.7, "lambda": 0.8,
               "eta": 0.3, "cash": 1.7, "error_mean": 0.0, "error_std": 1.0,
               "error_quantiles": [0.0] * 5, "mean_cost": 0.0, "mean_turnover": 0.0,
               "mean_holdings": [[0.0, 0.0], [0.0, 0.0]],
               "mean_turnover_by_date": [0.0, 0.0], "errors": [0.0, 1.0]}
    directory = save_artifacts(strategy, history, metrics, config, robust_state)
    assert (directory / "robust.eqx").exists()
    restored = eqx.tree_deserialise_leaves(directory / "robust.eqx", robust_state)
    for original, loaded in zip(jax.tree.leaves(robust_state), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(original, loaded)

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
    hedging_loss,
    initialize_strategy,
    make_steps,
    market_dimensions,
    save_artifacts,
    simulate_black_scholes,
    simulate_heston,
    strategy_apply,
)


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


def test_strategy_widths_output_and_batch_norm_modes():
    """Verify both paper architectures and that BN updates only in training mode."""

    for name, expected in (("black_scholes", (1, 1)), ("heston", (2, 2))):
        market = MarketConfig(name=name, n_time=3)
        n_assets, n_features = market_dimensions(market)
        assert (n_assets, n_features) == expected
        strategy = initialize_strategy(jax.random.PRNGKey(n_assets), market)
        features = jax.random.normal(jax.random.PRNGKey(8), (8, 3, n_features))
        holdings, trained = strategy_apply(strategy, features, StrategyConfig(), True)
        chex.assert_shape(holdings, (8, 3, n_assets))
        assert not np.array_equal(strategy.mean1, trained.mean1)
        _, inferred = strategy_apply(trained, features, StrategyConfig(), False)
        np.testing.assert_array_equal(trained.mean1, inferred.mean1)


def test_previous_holding_affects_later_decision():
    """Verify the recurrent previous-holding input can alter the following date's network output."""

    market = MarketConfig(n_time=2)
    strategy = initialize_strategy(jax.random.PRNGKey(18), market)
    features = jnp.full((4, 2, 1), jnp.log(100.0))
    baseline, _ = strategy_apply(strategy, features, StrategyConfig(), False)
    changed_bias = strategy.b3.at[0, 0].add(1.0)
    changed = eqx.tree_at(lambda candidate: candidate.b3, strategy, changed_bias)
    perturbed, _ = strategy_apply(changed, features, StrategyConfig(), False)
    assert not np.array_equal(baseline[:, 1], perturbed[:, 1])


def test_zero_and_constant_strategy_accounting():
    """Verify zero gains/costs and that constant holdings incur only the initial trade cost."""

    market = MarketConfig(n_time=3, horizon=3 / 365)
    strategy = initialize_strategy(jax.random.PRNGKey(3), market)
    strategy = eqx.tree_at(lambda s: (s.w1, s.b1, s.w2, s.b2, s.w3, s.b3), strategy,
                           tuple(jnp.zeros_like(x) for x in (strategy.w1, strategy.b1, strategy.w2,
                                                            strategy.b2, strategy.w3, strategy.b3)))
    prices = jnp.asarray([[[100.0], [101.0], [99.0], [102.0]]])
    features = jnp.log(prices[:, :-1])
    _, (_, gains, costs, _, holdings), _ = hedging_loss(strategy, prices, features, 0.01, StrategyConfig(), False)
    np.testing.assert_array_equal(holdings, 0.0)
    np.testing.assert_array_equal(gains, 0.0)
    np.testing.assert_array_equal(costs, 0.0)
    strategy = eqx.tree_at(lambda s: s.b3, strategy, jnp.ones_like(strategy.b3))
    _, (_, gains, costs, _, holdings), _ = hedging_loss(strategy, prices, features, 0.01, StrategyConfig(), False)
    np.testing.assert_allclose(holdings, 1.0)
    np.testing.assert_allclose(gains, 2.0)
    np.testing.assert_allclose(costs, 1.0)


def test_costs_are_exactly_zero_or_nonnegative():
    """Verify epsilon zero removes costs exactly and positive epsilon produces nonnegative costs."""

    market = MarketConfig(n_time=3, horizon=3 / 365)
    strategy = initialize_strategy(jax.random.PRNGKey(13), market)
    prices, features = simulate_black_scholes(jax.random.PRNGKey(14), market, 16)
    _, (_, _, zero_cost, _, _), _ = hedging_loss(strategy, prices, features, 0.0, StrategyConfig(), False)
    _, (_, _, positive_cost, _, _), _ = hedging_loss(strategy, prices, features, 0.01, StrategyConfig(), False)
    np.testing.assert_array_equal(zero_cost, 0.0)
    assert bool(jnp.all(positive_cost >= 0.0))
    assert bool(jnp.any(positive_cost > 0.0))


def test_jitted_optimizer_step_is_finite_and_updates_parameters():
    """Verify one top-level compiled Adam step returns finite values and changes trainable leaves."""

    market = MarketConfig(n_time=3, horizon=3 / 365)
    strategy = initialize_strategy(jax.random.PRNGKey(4), market)
    prices, features = simulate_black_scholes(jax.random.PRNGKey(5), market, 32)
    optimizer = optax.adam(0.005)
    state = optimizer.init(eqx.filter(strategy, eqx.is_array))
    step, _ = make_steps(optimizer, StrategyConfig(), 0.0)
    updated, _, loss, norm = step(strategy, state, prices, features)
    assert bool(jnp.isfinite(loss))
    assert bool(jnp.isfinite(norm))
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(strategy), jax.tree.leaves(updated)))


def test_short_fixed_batch_training_reduces_loss():
    """Verify fixed validation inference MSE falls after training on a separate fixed batch."""

    market = MarketConfig(n_time=3, horizon=3 / 365)
    strategy = initialize_strategy(jax.random.PRNGKey(6), market)
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
    strategy = initialize_strategy(jax.random.PRNGKey(15), config.market)
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

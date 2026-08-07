"""Run a small deterministic benchmark to verify nominal and WDRO learning."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from typing import Literal

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from deep_hedging import (
    Config,
    MarketConfig,
    Strategy,
    StrategyConfig,
    TrainingConfig,
    decode_trajectory,
    encode_trajectory,
    initialize_strategy,
    make_robust_steps,
    make_steps,
    pathwise_hedging_loss,
    simulate_black_scholes,
    strategy_apply,
    train,
)
from wdro import RobustConfig, RobustState

Array = jax.Array
Mode = Literal["nominal", "entropic", "relaxed_kl_cvar"]


@chex.dataclass
class LearningBenchmarkConfig:
    """Configure the small learning benchmark; fields control its shared budget and output."""

    steps: int = 500
    mode: Literal["all", "nominal", "entropic", "relaxed_kl_cvar"] = "all"
    batch_size: int = 64
    validation_size: int = 2_048
    evaluation_size: int = 10_000
    validation_frequency: int = 25
    learning_rate: float = 0.001
    proposal_samples: int = 8
    proposal_scale: float = 0.05
    ula_steps: int = 0
    ula_step_size: float = 0.0025
    rho: float = 0.075
    kl_epsilon: float = 0.1
    alpha: float = 0.95
    seed: int = 0
    output_dir: str = "benchmark/learning_smoke_k0"


def experiment_config(mode: Mode, benchmark: LearningBenchmarkConfig) -> Config:
    """Build one fair model configuration; inputs are mode/benchmark and output is Config."""

    market = MarketConfig(n_time=5, horizon=5.0 / 365.0)
    training = TrainingConfig(
        steps=benchmark.steps,
        batch_size=benchmark.batch_size,
        learning_rate=benchmark.learning_rate,
        validation_size=benchmark.validation_size,
        validation_frequency=benchmark.validation_frequency,
        seed=benchmark.seed,
    )
    robust = RobustConfig(
        enabled=mode != "nominal",
        objective="entropic" if mode == "nominal" else mode,
        rho=benchmark.rho,
        kl_epsilon=benchmark.kl_epsilon,
        tau=0.0,
        alpha=benchmark.alpha,
        proposal_samples=benchmark.proposal_samples,
        proposal_scale=benchmark.proposal_scale,
        ula_steps=benchmark.ula_steps,
        ula_step_size=benchmark.ula_step_size,
        ula_gradient_clip=10.0,
        initial_lambda=1.0,
        initial_eta=0.0,
    )
    return Config(market=market, training=training, robust=robust, epsilon=0.0, alpha=benchmark.alpha)


def initial_state_and_data(
    config: Config,
) -> tuple[Strategy, RobustState | None, Array, Array, Array, Array]:
    """Reproduce train initialization/data; input is Config, outputs model/dual/validation paths/proposal key."""

    key = jax.random.PRNGKey(config.training.seed)
    _, init_key, standardization_key, validation_key, validation_proposal_key = jax.random.split(key, 5)
    strategy = initialize_strategy(init_key, config.market)
    validation_prices, validation_features = simulate_black_scholes(
        validation_key, config.market, config.training.validation_size)
    robust_state = None
    if config.robust.enabled:
        standardization_prices, _ = simulate_black_scholes(
            standardization_key, config.market, config.training.validation_size)
        standardization_coordinates = encode_trajectory(
            standardization_prices, None, config.market)
        dimension = standardization_coordinates.shape[1]
        chex.assert_shape(
            standardization_coordinates, (config.training.validation_size, dimension))
        location = jnp.mean(standardization_coordinates, axis=0)
        chex.assert_shape(location, (dimension,))
        scale = jnp.maximum(jnp.std(standardization_coordinates, axis=0), 1e-3)
        chex.assert_shape(scale, (dimension,))
        raw_lambda = jnp.log(jnp.expm1(jnp.asarray(config.robust.initial_lambda)))
        chex.assert_shape(raw_lambda, ())
        eta = jnp.asarray(config.robust.initial_eta)
        chex.assert_shape(eta, ())
        robust_state = RobustState(raw_lambda, eta, location, scale)
    return (strategy, robust_state, validation_prices, validation_features,
            validation_proposal_key, validation_key)


def calibrate(strategy: Strategy, features: Array, strategy_config: StrategyConfig) -> Strategy:
    """Calibrate batch normalization on fixed features; inputs are model/features/config, output is model."""

    calibration = dataclasses.replace(strategy_config, bn_momentum=0.0)
    _, calibrated = strategy_apply(strategy, features, calibration, True)
    return calibrated


def common_metrics(
    strategy: Strategy,
    prices: Array,
    features: Array,
    config: Config,
) -> dict[str, float]:
    """Compute common unpriced distribution metrics; inputs are model/test paths/config, output is metrics."""

    strategy = eqx.tree_at(
        lambda candidate: candidate.cash, strategy, jnp.zeros_like(strategy.cash))
    losses, _, _ = pathwise_hedging_loss(
        strategy, prices, features, config.epsilon, config.strategy, False,
        config.market.strike)
    batch = losses.shape[0]
    chex.assert_shape(losses, (batch,))
    losses_np = np.asarray(losses)
    quantile_95 = float(np.quantile(losses_np, 0.95))
    quantile_99 = float(np.quantile(losses_np, 0.99))
    cvar_95 = float(np.mean(losses_np[losses_np >= quantile_95]))
    cvar_99 = float(np.mean(losses_np[losses_np >= quantile_99]))
    median = float(np.median(losses_np))
    return {
        "mean": float(np.mean(losses_np)),
        "median": median,
        "var_95": quantile_95,
        "cvar_95": cvar_95,
        "cvar_99": cvar_99,
        "iqr": float(np.quantile(losses_np, 0.75) - np.quantile(losses_np, 0.25)),
        "mad": float(np.median(np.abs(losses_np - median))),
        "stop_loss_0": float(np.mean(np.maximum(losses_np, 0.0))),
    }


def native_evaluation(
    mode: Mode,
    strategy: Strategy,
    robust_state: RobustState | None,
    prices: Array,
    features: Array,
    proposal_key: Array,
    config: Config,
) -> tuple[float, float | None, float | None]:
    """Evaluate one native objective and active-tail fraction; inputs are states/data/config, outputs scalars."""

    optimizer = optax.adam(config.training.learning_rate)
    if mode == "nominal":
        _, evaluate = make_steps(
            optimizer, config.strategy, config.epsilon, config.alpha,
            config.market.strike)
        objective, _ = evaluate(strategy, prices, features)
        return float(objective), None, None
    if robust_state is None:
        raise ValueError("robust mode requires RobustState")
    _, evaluate = make_robust_steps(
        optimizer, config.market, config.strategy, config.robust, config.epsilon)
    objective, diagnostics = evaluate(
        strategy, robust_state, prices, None, proposal_key)
    pathwise_losses, transport_cost, log_weights = diagnostics[:3]
    batch, n_proposals = pathwise_losses.shape
    chex.assert_shape(transport_cost, (batch, n_proposals))
    lambda_ = jax.nn.softplus(robust_state.raw_lambda)
    chex.assert_shape(lambda_, ())
    penalized = pathwise_losses - lambda_ * transport_cost
    chex.assert_shape(penalized, (batch, n_proposals))
    active_fraction = jnp.mean(penalized > robust_state.eta)
    chex.assert_shape(active_fraction, ())
    shifted_weights = jnp.exp(log_weights - jnp.max(log_weights, axis=1, keepdims=True))
    chex.assert_shape(shifted_weights, (batch, n_proposals))
    effective_sample_size = jnp.square(jnp.sum(shifted_weights, axis=1)) / jnp.sum(
        jnp.square(shifted_weights), axis=1)
    chex.assert_shape(effective_sample_size, (batch,))
    normalized_ess = jnp.mean(effective_sample_size) / n_proposals
    chex.assert_shape(normalized_ess, ())
    active_value = (
        float(active_fraction) if config.robust.objective == "relaxed_kl_cvar" else None)
    return float(objective), active_value, float(normalized_ess)


def strategy_distance(initial: Strategy, trained: Strategy) -> float:
    """Measure network parameter movement; inputs are initial/trained models, output is Euclidean distance."""

    initial_arrays = (initial.w1, initial.b1, initial.w2, initial.b2, initial.w3,
                      initial.b3, initial.gamma1, initial.beta1, initial.gamma2,
                      initial.beta2)
    trained_arrays = (trained.w1, trained.b1, trained.w2, trained.b2, trained.w3,
                      trained.b3, trained.gamma1, trained.beta1, trained.gamma2,
                      trained.beta2)
    squared = sum(float(jnp.sum(jnp.square(after - before)))
                  for before, after in zip(initial_arrays, trained_arrays))
    return float(np.sqrt(squared))


def run_mode(mode: Mode, benchmark: LearningBenchmarkConfig, directory: Path) -> dict:
    """Train and assess one benchmark mode; inputs are mode/config/output, output is result dictionary."""

    config = experiment_config(mode, benchmark)
    initial, robust_state, validation_prices, validation_features, proposal_key, _ = (
        initial_state_and_data(config))
    initial = calibrate(initial, validation_features, config.strategy)
    if mode == "relaxed_kl_cvar" and robust_state is not None:
        _, evaluate = make_robust_steps(
            optax.adam(config.training.learning_rate), config.market, config.strategy,
            config.robust, config.epsilon)
        _, diagnostics = evaluate(
            initial, robust_state, validation_prices, None, proposal_key)
        pathwise_losses, transport_cost = diagnostics[:2]
        lambda_ = jax.nn.softplus(robust_state.raw_lambda)
        chex.assert_shape(lambda_, ())
        pilot_eta = jnp.quantile(pathwise_losses - lambda_ * transport_cost, 0.95)
        chex.assert_shape(pilot_eta, ())
        config.robust.initial_eta = float(pilot_eta)
        robust_state = eqx.tree_at(
            lambda candidate: candidate.eta, robust_state, pilot_eta)
    initial_native, initial_active, initial_ess = native_evaluation(
        mode, initial, robust_state, validation_prices, validation_features,
        proposal_key, config)
    initial_common = common_metrics(
        initial, validation_prices, validation_features, config)
    trained, trained_robust, history = train(config)
    trained_native, trained_active, trained_ess = native_evaluation(
        mode, trained, trained_robust, validation_prices, validation_features,
        proposal_key, config)
    trained_common = common_metrics(
        trained, validation_prices, validation_features, config)
    test_prices, test_features = simulate_black_scholes(
        jax.random.PRNGKey(100_000 + benchmark.seed), config.market,
        benchmark.evaluation_size)
    test_common = common_metrics(trained, test_prices, test_features, config)
    native_improvement = (initial_native - trained_native) / max(abs(initial_native), 1e-6)
    cvar_improvement = (
        initial_common["cvar_95"] - trained_common["cvar_95"]
    ) / max(abs(initial_common["cvar_95"]), 1e-6)
    stop_improvement = (
        initial_common["stop_loss_0"] - trained_common["stop_loss_0"]
    ) / max(abs(initial_common["stop_loss_0"]), 1e-6)
    grad_norms = np.asarray(history["grad_norm"])
    finite_nonzero_grad_fraction = float(np.mean(
        np.isfinite(grad_norms) & (grad_norms > 1e-8)))
    objective_learning = bool(
        np.isfinite(initial_native)
        and np.isfinite(trained_native)
        and native_improvement >= 0.02
        and finite_nonzero_grad_fraction >= 0.90
        and strategy_distance(initial, trained) > 1e-4
    )
    risk_improving = bool(cvar_improvement >= 0.02 or stop_improvement >= 0.02)
    tail_calibrated = True
    if mode == "relaxed_kl_cvar":
        if trained_active is None:
            raise ValueError("relaxed-KL-CVaR evaluation must return an active-tail fraction")
        tail_calibrated = 0.02 <= trained_active <= 0.15
    proposal_healthy = bool(
        benchmark.ula_steps == 0
        or (trained_ess is not None and np.isfinite(trained_ess) and trained_ess >= 0.20))
    learned = objective_learning and risk_improving and tail_calibrated and proposal_healthy
    result = {
        "mode": mode,
        "learned": learned,
        "objective_learning": objective_learning,
        "risk_improving": risk_improving,
        "tail_calibrated": tail_calibrated,
        "proposal_healthy": proposal_healthy,
        "initial_native_objective": initial_native,
        "trained_native_objective": trained_native,
        "native_improvement_fraction": native_improvement,
        "validation_cvar_improvement_fraction": cvar_improvement,
        "validation_stop_loss_improvement_fraction": stop_improvement,
        "initial_active_tail_fraction": initial_active,
        "trained_active_tail_fraction": trained_active,
        "initial_normalized_ess": initial_ess,
        "trained_normalized_ess": trained_ess,
        "finite_nonzero_grad_fraction": finite_nonzero_grad_fraction,
        "strategy_parameter_distance": strategy_distance(initial, trained),
        "initial_validation": initial_common,
        "trained_validation": trained_common,
        "trained_test": test_common,
        "lambda": None if trained_robust is None else float(
            jax.nn.softplus(trained_robust.raw_lambda)),
        "eta": None if trained_robust is None else float(trained_robust.eta),
        "history": history,
        "config": dataclasses.asdict(config),
    }
    mode_directory = directory / mode
    mode_directory.mkdir(parents=True, exist_ok=True)
    (mode_directory / "result.json").write_text(json.dumps(result, indent=2))
    eqx.tree_serialise_leaves(mode_directory / "model.eqx", trained)
    if trained_robust is not None:
        eqx.tree_serialise_leaves(mode_directory / "robust.eqx", trained_robust)
    active_text = "n/a" if trained_active is None else f"{trained_active:.3%}"
    ess_text = "n/a" if trained_ess is None else f"{trained_ess:.3%}"
    print(f"RESULT mode={mode} learned={learned} objective_learning={objective_learning} "
          f"risk_improving={risk_improving} tail_calibrated={tail_calibrated} "
          f"proposal_healthy={proposal_healthy} "
          f"native_change={native_improvement:.3%} "
          f"cvar_change={cvar_improvement:.3%} active_tail={active_text} ess={ess_text}")
    return result


def main(benchmark: LearningBenchmarkConfig) -> None:
    """Run all three learning checks; input is benchmark config, output is saved benchmark artifacts."""

    directory = Path(benchmark.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "benchmark_config.json").write_text(
        json.dumps(dataclasses.asdict(benchmark), indent=2))
    modes: tuple[Mode, ...] = (
        ("nominal", "entropic", "relaxed_kl_cvar")
        if benchmark.mode == "all" else (benchmark.mode,))
    results = [run_mode(mode, benchmark, directory) for mode in modes]
    fields = [
        "mode", "learned", "objective_learning", "risk_improving", "tail_calibrated",
        "proposal_healthy",
        "initial_native_objective", "trained_native_objective",
        "native_improvement_fraction", "validation_cvar_improvement_fraction",
        "validation_stop_loss_improvement_fraction", "trained_active_tail_fraction",
        "trained_normalized_ess",
        "finite_nonzero_grad_fraction", "strategy_parameter_distance", "lambda", "eta",
    ]
    with (directory / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: result[field] for field in fields} for result in results)
    (directory / "results.json").write_text(json.dumps(results, indent=2))
    robust_results = [result for result in results if result["mode"] != "nominal"]
    robust_pass = bool(robust_results) and all(result["learned"] for result in robust_results)
    print(f"ROBUST_PASS={robust_pass}")


if __name__ == "__main__":
    main(tyro.cli(LearningBenchmarkConfig))

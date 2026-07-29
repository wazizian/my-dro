"""Sweep harness for the LSV sample-size study (see lsv.md)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from deep_hedging import (
    Config,
    EvaluationConfig,
    MarketConfig,
    StrategyConfig,
    TrainingConfig,
    pathwise_hedging_loss,
    simulate_market,
    train,
)
from wdro import RobustConfig

Array = jax.Array
Dataset = tuple[Array, Array, Array | None]

METHODS = ("cvar", "cvar_of_average", "relaxed_kl_wdro")
N_GRID = (100, 250, 500, 750, 1000)
SEEDS = (0, 1, 2)
N_MAX = 1000
N_TEST = 20_000
K_FOLDS = 3
STEPS = 2000
BATCH_SIZE = 64
CANONICAL_ALPHA = 0.95

# In-distribution / 3-tier leverage-shift severity ladder, per lsv.md S1/S4. weak/medium/hard are
# 1.5x/2x/3x multiples of the in_dist (leverage_beta, leverage_kappa) offset from zero.
LEVERAGE_PARAMS = {
    "in_dist": {"leverage_beta": -0.3, "leverage_kappa": 0.5},
    "weak": {"leverage_beta": -0.45, "leverage_kappa": 0.75},
    "medium": {"leverage_beta": -0.6, "leverage_kappa": 1.0},
    "hard": {"leverage_beta": -0.9, "leverage_kappa": 1.5},
}
SHIFT_TIERS = ("weak", "medium", "hard")

# tag -> fixed integer offset for deterministic, cache-stable PRNG derivation.
_TAG_OFFSETS = {"train_pool": 0, "test_indist": 1, "test_weak": 2, "test_medium": 3, "test_hard": 4}


def lsv_market(variant: Literal["in_dist", "weak", "medium", "hard"]) -> MarketConfig:
    """Build the LSV MarketConfig for a leverage variant; input variant name, output MarketConfig."""

    return MarketConfig(name="lsv", n_time=100, horizon=100.0 / 365.0, dtype="float32",
                        n_substeps=10, **LEVERAGE_PARAMS[variant])


def _cache_file(cache_dir: Path, seed: int, tag: str, n_paths: int) -> Path:
    """Build the cache filename for one (seed, tag, n_paths) dataset; inputs directory/seed/tag/n, output path."""

    return cache_dir / f"seed{seed}_{tag}_n{n_paths}.npz"


def load_or_simulate(cache_dir: Path, seed: int, tag: str, market: MarketConfig, n_paths: int) -> Dataset:
    """Load a cached dataset or simulate and cache it; inputs cache location/config, output dataset triple.

    The cache filename encodes n_paths, so a smoke test run with a small --n-test/--n-max cannot
    silently be reused by a later full run with different sizes pointed at the same --cache-dir.
    """

    path = _cache_file(cache_dir, seed, tag, n_paths)
    if path.exists():
        data = np.load(path)
        prices = jnp.asarray(data["prices"])
        features = jnp.asarray(data["features"])
        variance = jnp.asarray(data["variance"]) if "variance" in data else None
        assert prices.shape[0] == n_paths, f"cache {path} has {prices.shape[0]} paths, expected {n_paths}"
        return prices, features, variance
    key = jax.random.fold_in(jax.random.PRNGKey(seed), _TAG_OFFSETS[tag])
    prices, features, variance = simulate_market(key, market, n_paths)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {"prices": np.asarray(prices), "features": np.asarray(features)}
    if variance is not None:
        payload["variance"] = np.asarray(variance)
    np.savez(path, **payload)
    return prices, features, variance


def prefix(dataset: Dataset, n: int) -> Dataset:
    """Take the first n paths of a dataset (nested-prefix trick); inputs dataset/n, output sliced dataset."""

    prices, features, variance = dataset
    return prices[:n], features[:n], None if variance is None else variance[:n]


def grid_for(method: str) -> list[dict[str, float]]:
    """Return the hyperparameter grid for a method, per lsv.md S3; input method name, output config list."""

    if method == "cvar":
        return [{"learning_rate": lr, "alpha": alpha}
                for lr in (0.002, 0.005, 0.01) for alpha in (0.90, 0.95, 0.99)]
    if method == "cvar_of_average":
        return [{"learning_rate": lr, "proposal_scale": scale, "alpha": alpha}
                for lr in (0.002, 0.01) for scale in (0.05, 0.2) for alpha in (0.90, 0.99)]
    if method == "relaxed_kl_wdro":
        return [{"learning_rate": lr, "rho": rho, "alpha": alpha}
                for lr in (0.002, 0.01) for rho in (0.03, 0.3) for alpha in (0.90, 0.99)]
    raise ValueError(f"unknown method: {method}")


def method_config(method: str, hp: dict[str, float], market: MarketConfig, steps: int, seed: int) -> Config:
    """Build a full Config for one method/hyperparameter combo; inputs method/hp/market, output Config."""

    training = TrainingConfig(steps=steps, batch_size=BATCH_SIZE, learning_rate=hp["learning_rate"],
                              validation_frequency=max(steps // 10, 1), seed=seed)
    evaluation = EvaluationConfig()
    if method == "cvar":
        robust = RobustConfig(enabled=False)
    elif method == "cvar_of_average":
        robust = RobustConfig(enabled=True, objective="plain_cvar_of_average", proposal_samples=32,
                              proposal_scale=hp["proposal_scale"], alpha=hp["alpha"])
    elif method == "relaxed_kl_wdro":
        robust = RobustConfig(enabled=True, objective="relaxed_kl_cvar", proposal_samples=32,
                              kl_epsilon=0.03, ula_steps=0, cost_power=2, rho=hp["rho"], alpha=hp["alpha"])
    else:
        raise ValueError(f"unknown method: {method}")
    return Config(market=market, strategy=StrategyConfig(), training=training, evaluation=evaluation,
                 robust=robust, epsilon=0.0, alpha=hp["alpha"])


def empirical_cvar95(losses: np.ndarray) -> float:
    """Compute the canonical empirical CVaR at alpha=0.95; input a loss array, output a scalar."""

    threshold = np.quantile(losses, CANONICAL_ALPHA)
    return float(np.mean(losses[losses >= threshold]))


def evaluate_cvar95(strategy, strategy_config, market: MarketConfig, epsilon: float, dataset: Dataset) -> float:
    """Evaluate the canonical CVaR-95 of a calibrated strategy on a dataset; inputs model/data, output scalar."""

    prices, features, _ = dataset
    losses, _, _ = pathwise_hedging_loss(strategy, prices, features, epsilon, strategy_config, False, market.strike)
    return empirical_cvar95(np.asarray(losses))


def kfold_split(n: int, k: int, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Build train/val index masks for one fold; inputs pool size/fold count/index, output masks.

    k == 1 is a special case: a single deterministic 80/20 train/val split (last 20% of the
    prefix is validation), not a fold rotation -- there is only ever fold 0. k >= 2 is the
    original contiguous k-way rotation.
    """

    if k == 1:
        n_val = max(1, int(round(0.2 * n)))
        val_mask = np.zeros(n, dtype=bool)
        val_mask[n - n_val:] = True
        return ~val_mask, val_mask
    bounds = np.linspace(0, n, k + 1).astype(int)
    val_mask = np.zeros(n, dtype=bool)
    val_mask[bounds[fold]:bounds[fold + 1]] = True
    return ~val_mask, val_mask


def kfold_score(method: str, hp: dict[str, float], n: int, seed: int, pool: Dataset,
                market: MarketConfig, k: int, steps: int) -> tuple[float, list, Config]:
    """Compute mean k-fold validation CVaR-95 for one hyperparameter combo; inputs configs/data,
    output (mean score, per-fold trained strategies, the Config used to train them)."""

    prices, features, variance = prefix(pool, n)
    config = method_config(method, hp, market, steps, seed)
    scores = []
    strategies = []
    for fold in range(k):
        train_mask, val_mask = kfold_split(n, k, fold)
        train_ds = (prices[train_mask], features[train_mask], None if variance is None else variance[train_mask])
        val_ds = (prices[val_mask], features[val_mask], None if variance is None else variance[val_mask])
        strategy, _, _ = train(config, train_dataset=train_ds, validation_dataset=val_ds)
        scores.append(evaluate_cvar95(strategy, config.strategy, market, config.epsilon, val_ds))
        strategies.append(strategy)
    return float(np.mean(scores)), strategies, config


def run_one(method: str, n: int, seed: int, cache_dir: Path, k: int = K_FOLDS, steps: int = STEPS,
           n_max: int = N_MAX, n_test: int = N_TEST, retrain: bool = True) -> dict:
    """Tune, (re)fit, and test-evaluate one (method, n, seed) cell; inputs cell spec, output a results row.

    retrain=False skips the separate full-n refit and instead reuses the winning hyperparameter
    combo's already-trained model directly -- only valid with k=1 (a single train/val split), since
    with k>1 there'd be k candidate models per combo and no principled single choice among them.
    """

    if not retrain and k != 1:
        raise ValueError("retrain=False requires k=1 (single train/val split); ambiguous with k>1 folds")

    market_indist = lsv_market("in_dist")
    shift_markets = {tier: lsv_market(tier) for tier in SHIFT_TIERS}
    train_pool = load_or_simulate(cache_dir, seed, "train_pool", market_indist, n_max)
    test_indist = load_or_simulate(cache_dir, seed, "test_indist", market_indist, n_test)
    shift_test_sets = {tier: load_or_simulate(cache_dir, seed, f"test_{tier}", shift_markets[tier], n_test)
                       for tier in SHIFT_TIERS}

    grid = grid_for(method)
    scored = [(hp, *kfold_score(method, hp, n, seed, train_pool, market_indist, k, steps)) for hp in grid]
    best_hp, best_score, best_strategies, best_config = min(scored, key=lambda item: item[1])

    if retrain:
        full_ds = prefix(train_pool, n)
        config = method_config(method, best_hp, market_indist, steps, seed)
        # Refit checkpoint selection uses the training set itself as "validation" (no separate
        # refit-validation split exists in lsv.md S3) -- the held-out test sets must stay untouched
        # until final evaluation below. See lsv_stage3_notes.md choice #1.
        strategy, _, _ = train(config, train_dataset=full_ds, validation_dataset=full_ds)
    else:
        # Reuse the winning combo's single-split model as-is; it was trained on 80% of the n-path
        # pool, not the full n, since no separate refit is performed.
        strategy, config = best_strategies[0], best_config

    row = {"method": method, "n": n, "seed": seed, "kfold_val_cvar95": best_score,
          "test_cvar95_indist": evaluate_cvar95(strategy, config.strategy, market_indist, config.epsilon, test_indist)}
    for tier in SHIFT_TIERS:
        row[f"test_cvar95_{tier}"] = evaluate_cvar95(
            strategy, config.strategy, shift_markets[tier], config.epsilon, shift_test_sets[tier])
    row.update({f"hp_{key}": value for key, value in best_hp.items()})
    return row


CSV_FIELDNAMES = ["method", "n", "seed", "kfold_val_cvar95", "test_cvar95_indist",
                  "test_cvar95_weak", "test_cvar95_medium", "test_cvar95_hard",
                  "hp_learning_rate", "hp_alpha", "hp_proposal_scale", "hp_rho"]


def run_sweep(methods, n_values, seeds, cache_dir: Path, output_csv: Path, k: int = K_FOLDS,
             steps: int = STEPS, n_max: int = N_MAX, n_test: int = N_TEST, retrain: bool = True) -> None:
    """Run the full method x n x seed sweep, writing each row to CSV as soon as it's ready; inputs
    sweep spec, output none. Incremental writes mean a crash/interrupt only loses the in-flight
    cell, not already-finished cells -- important since a full sweep runs for hours."""

    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for method in methods:
            for n in n_values:
                for seed in seeds:
                    print(f"running method={method} n={n} seed={seed}")
                    row = run_one(method, n, seed, cache_dir, k, steps, n_max, n_test, retrain)
                    writer.writerow(row)
                    handle.flush()


def main() -> None:
    """Run the sweep from the CLI; inputs argv flags, output printed progress and a CSV file."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--n-values", nargs="+", type=int, default=list(N_GRID))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--k", type=int, default=K_FOLDS)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--n-max", type=int, default=N_MAX)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--cache-dir", type=Path, default=Path("lsv_cache"))
    parser.add_argument("--output", type=Path, default=Path("lsv_sweep_results.csv"))
    parser.add_argument("--retrain", dest="retrain", action="store_true", default=True,
                       help="refit on the full n-path pool after tuning (default)")
    parser.add_argument("--no-retrain", dest="retrain", action="store_false",
                       help="skip the full-n refit; reuse the winning hyperparameter combo's "
                            "already-trained model directly (requires --k 1)")
    args = parser.parse_args()
    run_sweep(args.methods, args.n_values, args.seeds, args.cache_dir, args.output,
             args.k, args.steps, args.n_max, args.n_test, args.retrain)


if __name__ == "__main__":
    main()

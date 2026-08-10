"""Sweep harness for the LSV sample-size study (see lsv.md)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax

from deep_hedging import (
    Config,
    EvaluationConfig,
    MarketConfig,
    StrategyConfig,
    TrainingConfig,
    encode_trajectory,
    initialize_strategy,
    make_robust_steps,
    pathwise_hedging_loss,
    simulate_market,
    train,
)
from wdro import RobustConfig, RobustState

Array = jax.Array
Dataset = tuple[Array, Array, Array | None]

METHODS = ("cvar", "relaxed_kl_wdro", "relaxed_kl_wdro_convex")

# Strike grid + lambda_ij training mode for the convex-order penalty (relaxed_kl_wdro_convex),
# per probe evidence: wide strikes clearly beat default/tight/atm-only, and trainable lambda_ij
# clearly beats a fixed penalty weight (see conversation preceding this sweep design).
CONVEX_ORDER_STRIKES = (80.0, 100.0, 120.0)
N_GRID = (100, 250, 500, 750, 1000)
SEEDS = (0, 1, 2)
N_MAX = 1000
N_TEST = 20_000
K_FOLDS = 4
STEPS = 500
BATCH_SIZE = 64
CANONICAL_ALPHA = 0.95
# GPU-study additions: proportional transaction cost (10bps, standard equity-option friction
# magnitude), a fixed (not swept) latent memory channel for the recurrent hedging policy (this
# implicitly selects the GRU+LayerNorm architecture over the legacy BN+MLP one -- see
# deep_hedging.py's initialize_strategy/strategy_apply), and a grad-norm clipping safety net
# (defense in depth alongside the GRU's gating/LayerNorm, not a substitute for it).
EPSILON = 0.001
LATENT_DIM = 4
GRAD_CLIP_NORM = 1.0

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


def lsv_market(variant: Literal["in_dist", "weak", "medium", "hard"], n_time: int = 100) -> MarketConfig:
    """Build the LSV MarketConfig for a leverage variant; input variant name/n_time, output MarketConfig.

    n_time also sets horizon (1 trading day per step, n_time/365 years) so the two stay
    consistent -- overriding n_time is mainly for fast smoke/integration tests, not the study.
    """

    return MarketConfig(name="lsv", n_time=n_time, horizon=n_time / 365.0, dtype="float32",
                        n_substeps=10, **LEVERAGE_PARAMS[variant])


def _cache_file(cache_dir: Path, seed: int, tag: str, n_paths: int, n_time: int) -> Path:
    """Build the cache filename for one (seed, tag, n_paths, n_time) dataset; inputs directory/seed/tag/n/n_time, output path."""

    return cache_dir / f"seed{seed}_{tag}_n{n_paths}_t{n_time}.npz"


def load_or_simulate(cache_dir: Path, seed: int, tag: str, market: MarketConfig, n_paths: int) -> Dataset:
    """Load a cached dataset or simulate and cache it; inputs cache location/config, output dataset triple.

    The cache filename encodes n_paths and n_time, so a smoke test run with a small --n-test/--n-max
    or a different --n-time cannot silently be reused by a later run pointed at the same --cache-dir.
    """

    path = _cache_file(cache_dir, seed, tag, n_paths, market.n_time)
    if path.exists():
        data = np.load(path)
        prices = jnp.asarray(data["prices"])
        features = jnp.asarray(data["features"])
        variance = jnp.asarray(data["variance"]) if "variance" in data else None
        assert prices.shape[0] == n_paths, f"cache {path} has {prices.shape[0]} paths, expected {n_paths}"
        assert prices.shape[1] == market.n_time + 1, (
            f"cache {path} has n_time {prices.shape[1] - 1}, expected {market.n_time}")
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


# rho scaled as c / sqrt(n) (standard Wasserstein-radius shrinkage rate) rather than a fixed
# grid -- 4 constants, log-spaced, tuned per-n instead of reusing the same raw rho across n.
RHO_CONSTANTS = (0.3, 1.0, 3.0, 10.0)


def grid_for(method: str, n: int) -> list[dict[str, float]]:
    """Return the hyperparameter grid for a method at pool size n; inputs method/n, output config list."""

    if method == "cvar":
        return [{"learning_rate": lr, "alpha": CANONICAL_ALPHA} for lr in (0.002, 0.005, 0.01)]
    if method == "cvar_of_average":
        return [{"learning_rate": lr, "proposal_scale": scale, "alpha": CANONICAL_ALPHA}
                for lr in (0.002, 0.01) for scale in (0.05, 0.2)]
    if method in ("relaxed_kl_wdro", "relaxed_kl_wdro_convex"):
        return [{"learning_rate": lr, "rho": c / np.sqrt(n), "alpha": CANONICAL_ALPHA}
                for lr in (0.002, 0.01) for c in RHO_CONSTANTS]
    raise ValueError(f"unknown method: {method}")


def method_config(method: str, hp: dict[str, float], market: MarketConfig, steps: int, seed: int) -> Config:
    """Build a full Config for one method/hyperparameter combo; inputs method/hp/market, output Config."""

    training = TrainingConfig(steps=steps, batch_size=BATCH_SIZE, learning_rate=hp["learning_rate"],
                              validation_frequency=max(steps // 10, 1), seed=seed,
                              grad_clip_norm=GRAD_CLIP_NORM)
    evaluation = EvaluationConfig()
    if method == "cvar":
        robust = RobustConfig(enabled=False)
    elif method == "cvar_of_average":
        robust = RobustConfig(enabled=True, objective="plain_cvar_of_average", proposal_samples=32,
                              proposal_scale=hp["proposal_scale"], alpha=hp["alpha"])
    elif method == "relaxed_kl_wdro":
        robust = RobustConfig(enabled=True, objective="relaxed_kl_cvar", proposal_samples=32,
                              kl_epsilon=0.03, ula_steps=0, cost_power=2, rho=hp["rho"], alpha=hp["alpha"])
    elif method == "relaxed_kl_wdro_convex":
        robust = RobustConfig(enabled=True, objective="relaxed_kl_cvar", proposal_samples=32,
                              kl_epsilon=0.03, ula_steps=0, cost_power=2, rho=hp["rho"], alpha=hp["alpha"],
                              enforce_convex_order=True, convex_order_strikes=CONVEX_ORDER_STRIKES,
                              convex_order_lambda_init=1.0, fix_convex_order_lambda=False)
    else:
        raise ValueError(f"unknown method: {method}")
    return Config(market=market, strategy=StrategyConfig(latent_dim=LATENT_DIM), training=training,
                 evaluation=evaluation, robust=robust, epsilon=EPSILON, alpha=hp["alpha"])


# Methods whose dual has a trainable CVaR threshold (RobustState.eta) that benefits from a
# pilot warm start -- cold-starting at RobustConfig.initial_eta=0.0 forces eta to crawl toward
# the true dual-optimal value at a roughly lr-capped rate per step, wasting a large fraction of
# a short training budget just getting eta into the right neighborhood (see lsv_stage3_notes.md
# convergence probe: eta was still drifting ~linearly after 600 steps, never having plateaued).
ETA_WARM_START_METHODS = ("relaxed_kl_wdro", "relaxed_kl_wdro_convex", "cvar_of_average")
PILOT_BATCH_SIZE = 256


def robust_scale(values: Array) -> Array:
    """Floor-guarded std, used as a lambda-init scale estimate; input any array, output positive scalar."""

    return jnp.maximum(jnp.std(values), 1e-6)


def pilot_warm_start(method: str, config: Config, market: MarketConfig, seed: int, dataset: Dataset) -> None:
    """Warm-start a dual's lambda/lambda_ij/eta from an untrained network's pilot batch; mutates config.robust.

    Each penalty multiplier is set so its penalty term starts at the same scale as the pathwise
    loss f it's subtracted from/added to -- initial_lambda so that lambda * transport_cost is
    comparable to f (otherwise the softplus(raw_lambda) gradient starts dominated or negligible
    relative to f's own scale), and convex_order_lambda_init likewise for lambda_ij * (call-price
    time-difference) against f. eta is then warm-started to the alpha-quantile of the same score
    the dual actually thresholds on, now computed under the freshly-scaled lambda(s) -- cold
    defaults (lambda=1, eta=0) otherwise force every dual variable to crawl from an arbitrary
    starting scale at a roughly lr-capped rate per step (see lsv_stage3_notes.md convergence
    probe: eta was still drifting ~linearly after 600 steps under the old lambda=1/eta=0 defaults).
    """

    if method not in ETA_WARM_START_METHODS:
        return
    prices, features, variance = dataset
    n_pilot = min(PILOT_BATCH_SIZE, prices.shape[0])
    prices, features, variance = prices[:n_pilot], features[:n_pilot], (
        None if variance is None else variance[:n_pilot])
    key = jax.random.PRNGKey(seed)
    init_key, proposal_key = jax.random.split(key)
    strategy = initialize_strategy(init_key, market, config.strategy)

    if method == "relaxed_kl_wdro_convex":
        raw_loss, _, _ = pathwise_hedging_loss(
            strategy, prices, features, config.epsilon, config.strategy, False, market.strike)
        stock_prices = prices[..., 0]
        strikes = jnp.asarray(config.robust.convex_order_strikes, dtype=stock_prices.dtype)
        calls = jax.nn.relu(stock_prices[:, :, None] - strikes[None, None, :])
        diff = calls[:, :-1, :] - calls[:, 1:, :]
        config.robust.convex_order_lambda_init = float(robust_scale(raw_loss) / robust_scale(diff))

    coordinates = encode_trajectory(prices, variance, market)
    location = jnp.mean(coordinates, axis=0)
    scale = jnp.maximum(jnp.std(coordinates, axis=0), 1e-3)
    n_strikes = len(config.robust.convex_order_strikes)
    raw_lambda_ij_init = jnp.log(jnp.expm1(jnp.asarray(config.robust.convex_order_lambda_init)))
    raw_lambda_ij = jnp.full((n_strikes, market.n_time), raw_lambda_ij_init)
    robust_state = RobustState(jnp.asarray(0.0), jnp.asarray(0.0), location, scale, raw_lambda_ij)
    _, evaluate = make_robust_steps(
        optax.adam(config.training.learning_rate), market, config.strategy, config.robust, config.epsilon)
    _, diagnostics = evaluate(strategy, robust_state, prices, variance, proposal_key)
    pathwise_losses, transport_cost, log_weights = diagnostics[:3]
    if method == "cvar_of_average":
        # plain_cvar_of_average_dual ignores lambda/transport_cost entirely -- only eta applies.
        weights = jax.nn.softmax(log_weights, axis=1)
        score = jnp.sum(weights * pathwise_losses, axis=1)
    else:
        config.robust.initial_lambda = float(robust_scale(pathwise_losses) / robust_scale(transport_cost))
        lambda_ = jax.nn.softplus(jnp.asarray(config.robust.initial_lambda))
        score = pathwise_losses - lambda_ * transport_cost
    config.robust.initial_eta = float(jnp.quantile(score, config.robust.alpha))


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
        pilot_warm_start(method, config, market, seed, train_ds)
        strategy, _, _ = train(config, train_dataset=train_ds, validation_dataset=val_ds)
        scores.append(evaluate_cvar95(strategy, config.strategy, market, config.epsilon, val_ds))
        strategies.append(strategy)
    return float(np.mean(scores)), strategies, config


def run_one(method: str, n: int, seed: int, cache_dir: Path, k: int = K_FOLDS, steps: int = STEPS,
           n_max: int = N_MAX, n_test: int = N_TEST, retrain: bool = True, n_time: int = 100) -> dict:
    """Tune, (re)fit, and test-evaluate one (method, n, seed) cell; inputs cell spec, output a results row.

    retrain=False skips the separate full-n refit and instead reuses the winning hyperparameter
    combo's already-trained model directly -- only valid with k=1 (a single train/val split), since
    with k>1 there'd be k candidate models per combo and no principled single choice among them.
    n_time overrides the market's horizon -- only for fast smoke/integration tests, not the study.
    """

    if not retrain and k != 1:
        raise ValueError("retrain=False requires k=1 (single train/val split); ambiguous with k>1 folds")

    market_indist = lsv_market("in_dist", n_time)
    shift_markets = {tier: lsv_market(tier, n_time) for tier in SHIFT_TIERS}
    train_pool = load_or_simulate(cache_dir, seed, "train_pool", market_indist, n_max)
    test_indist = load_or_simulate(cache_dir, seed, "test_indist", market_indist, n_test)
    shift_test_sets = {tier: load_or_simulate(cache_dir, seed, f"test_{tier}", shift_markets[tier], n_test)
                       for tier in SHIFT_TIERS}

    grid = grid_for(method, n)
    scored = [(hp, *kfold_score(method, hp, n, seed, train_pool, market_indist, k, steps)) for hp in grid]
    best_hp, best_score, best_strategies, best_config = min(scored, key=lambda item: item[1])

    if retrain:
        full_ds = prefix(train_pool, n)
        config = method_config(method, best_hp, market_indist, steps, seed)
        pilot_warm_start(method, config, market_indist, seed, full_ds)
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
             steps: int = STEPS, n_max: int = N_MAX, n_test: int = N_TEST, retrain: bool = True,
             n_time: int = 100) -> None:
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
                    row = run_one(method, n, seed, cache_dir, k, steps, n_max, n_test, retrain, n_time)
                    writer.writerow(row)
                    handle.flush()


def precache(seeds, cache_dir: Path, n_max: int, n_test: int, n_time: int) -> None:
    """Simulate and cache every dataset a sweep will need, sequentially; inputs seeds/sizes, output none.

    Run this once, single-process, before launching multiple GPU-sharded sweep processes against
    the same --cache-dir: load_or_simulate only ever writes a cache file on a miss, so two
    processes racing to simulate the same (seed, tag, n) dataset concurrently could corrupt or
    duplicate the write. Pre-warming here means every later load_or_simulate call is a cache hit
    (read-only), so parallel processes never touch the filesystem for the same path at once.
    """

    market_indist = lsv_market("in_dist", n_time)
    shift_markets = {tier: lsv_market(tier, n_time) for tier in SHIFT_TIERS}
    for seed in seeds:
        print(f"precaching seed={seed}")
        load_or_simulate(cache_dir, seed, "train_pool", market_indist, n_max)
        load_or_simulate(cache_dir, seed, "test_indist", market_indist, n_test)
        for tier in SHIFT_TIERS:
            load_or_simulate(cache_dir, seed, f"test_{tier}", shift_markets[tier], n_test)


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
    parser.add_argument("--n-time", type=int, default=100,
                       help="market horizon in trading days (also sets horizon=n_time/365)")
    parser.add_argument("--cache-dir", type=Path, default=Path("lsv_cache"))
    parser.add_argument("--output", type=Path, default=Path("lsv_sweep_results.csv"))
    parser.add_argument("--retrain", dest="retrain", action="store_true", default=True,
                       help="refit on the full n-path pool after tuning (default)")
    parser.add_argument("--no-retrain", dest="retrain", action="store_false",
                       help="skip the full-n refit; reuse the winning hyperparameter combo's "
                            "already-trained model directly (requires --k 1)")
    parser.add_argument("--precache", action="store_true",
                       help="only simulate/cache all datasets for --seeds, then exit -- run this "
                            "once before launching multiple GPU-sharded sweep processes against "
                            "the same --cache-dir to avoid a concurrent-write race")
    args = parser.parse_args()
    if args.precache:
        precache(args.seeds, args.cache_dir, args.n_max, args.n_test, args.n_time)
        return
    run_sweep(args.methods, args.n_values, args.seeds, args.cache_dir, args.output,
             args.k, args.steps, args.n_max, args.n_test, args.retrain, args.n_time)


if __name__ == "__main__":
    main()

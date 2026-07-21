# LSV sample-size study — plan

Goal: measure how CVaR-95 test performance (in-dist and leverage-shifted) scales with
training sample size `n`, across 3 objectives, on a local-stochastic-vol (LSV) market
that is simulated close to its true SDE (fine sub-stepping) over a long, daily-hedge
horizon.

## 1. Market model: Heston + leverage

Extend `simulate_heston` with a leverage function `L(t, S)` multiplying the diffusion
coefficient of the stock leg, variance leg unchanged (still exact CIR):

```
dV_t = kappa (theta - V_t) dt + xi sqrt(V_t) dW^V_t                    (unchanged CIR)
dS_t = L(t, S_t) sqrt(V_t) S_t (rho dW^V_t + sqrt(1-rho^2) dW_perp_t)
```

`L == 1` recovers plain Heston, so this is a strict superset — existing Heston code path
stays valid as a regression check.

**Leverage function form**: CEV-style skew term times a mild time-growth term,
`L(t, S) = (S / S0)^beta * (1 + kappa_L * t)`.

- In-distribution / training params: `beta = -0.3`, `kappa_L = 0.5`.
- Shifted-test params (§4): `beta = -0.45`, `kappa_L = 0.75` — both magnitudes scaled
  ×1.5 (steeper negative skew, faster time-growth), same functional form. A fixed,
  single offset, not a grid.

**Discretization.** No closed form exists once `L` depends on `S`, so the stock leg is
Euler (`L` evaluated at the left endpoint of each fine substep); the variance leg keeps
the existing exact noncentral-chi-square CIR sampler (transition law doesn't depend on
`L`). To keep Euler error small, simulate on a **fine grid**: `n_substeps` (10) fine
steps per hedging step, i.e. total fine steps `= n_time * n_substeps`. The strategy only
observes/trades at the coarse `n_time` grid; fine intermediate points are integrated
over (for the piecewise-constant integrated-variance leg, same as current Heston code)
and then discarded/subsampled down to the `n_time+1` coarse observations.

**Horizon.** `n_time = 100` daily hedges (was 20/30/60 in prior report), `horizon =
100/365`. This is the "longer duration" axis; everything else (spot/strike/Heston
params) reuses `MarketConfig` defaults unless noted.

Config-wise this is an additive `MarketConfig` extension: `name="lsv"`, `leverage_beta`,
`leverage_kappa`, `n_substeps`. No changes needed to `StrategyConfig`/`Strategy` beyond
what Heston already needs (2 tradables: stock + variance swap, same as now).

## 2. Benchmarks (3 objectives)

- `cvar` — plain nominal Rockafellar-Uryasev CVaR-95 loss (`robust.enabled=False`,
  no WDRO dual), batch_size=64. Not WDRO-robust — the plain baseline.
- `cvar_of_average` — `RobustConfig(objective="plain_cvar_of_average")`.
- `relaxed_kl_wdro` — `RobustConfig(objective="relaxed_kl_cvar")`.

(A `big_batch` compute-matched-nominal control, à la prior reports'
`nominal_bigbatch` = nominal batch × `proposal_samples`, was considered and dropped:
at this study's small `n` — as low as 100 — a compute-matched batch of `64×32=2048`
means heavy resampling-with-replacement from a tiny pool, a degenerate comparison.
Out of scope here.)

All 3 already exist in `wdro.py`/`deep_hedging.py`'s nominal-loss path; no new
objective code needed, only wiring into the new sweep harness.

**Per-method robustness knob** (the thing tuned via k-fold, beyond step size/alpha):

| method             | robustness knob | fixed at |
|---|---|---|
| `cvar`              | *(none — plain nominal CVaR loss)* | `robust.enabled=False`, `batch_size=64` |
| `cvar_of_average`   | `proposal_scale` (averaging-neighborhood width; no `lambda`/`rho` in this dual) | `proposal_samples=32` |
| `relaxed_kl_wdro`   | `rho`                          | `kl_epsilon=0.03`, `ula_steps=0` (i.e. `k=0`, the config the prior report found best — reopening `k>0` is out of scope here), `cost_power=2`, `proposal_samples=32` |

## 3. Sample-size grid, splitting, tuning

`n ∈ {100, 250, 500, 750, 1000}` — number of *training paths* per run.

For each `n`:
- k-fold cross-validation directly on the `n` paths, `k = 3` (fold size ≈ 33%). For
  each fold rotation: train on the `k-1` folds, evaluate CVaR-95 on the held-out fold.
- **Hyperparameters tuned via k-fold mean val CVaR** (val metric always scored at the
  canonical `alpha=0.95`, regardless of the train-time `alpha` a config uses, so
  configs stay comparable):

  | method | grid |
  |---|---|
  | `cvar`            | `learning_rate ∈ {0.002, 0.005, 0.01}` × `alpha ∈ {0.90, 0.95, 0.99}` → 9 configs |
  | `cvar_of_average`  | `learning_rate ∈ {0.002, 0.01}` × `proposal_scale ∈ {0.05, 0.2}` × `alpha ∈ {0.90, 0.99}` → 8 configs |
  | `relaxed_kl_wdro`  | `learning_rate ∈ {0.002, 0.01}` × `rho ∈ {0.03, 0.3}` × `alpha ∈ {0.90, 0.99}` → 8 configs |

  Grids centered on `TrainingConfig`/`RobustConfig` defaults and ranges already
  explored in the Heston/BS report. `cvar_of_average`/`relaxed_kl_wdro` use 2
  values/axis (endpoints only) rather than 3 — compute-budget trim, see §7.
- **Fixed, not tuned**: batch size (`64` for all 3 methods) and iteration count
  (`TrainingConfig.steps` fixed across all `n`/methods/hyperparams — no early
  stopping, no budget scaling).
- After picking the best hyperparams from k-fold, **refit once on the full `n`
  training paths** (all folds pooled) with those hyperparams. This refit is what gets
  evaluated at test time.

## 4. Test-time evaluation

Two test metrics per (method, n, seed, hyperparam-selected-model):

1. **In-distribution test CVaR-95**: evaluate the refit model on a *fixed, large* test
   set of LSV paths, `n_test = 20_000` (well above `n_max = 1000`, in\_dist leverage
   params `beta=-0.3, kappa_L=0.5`). Large enough that test CVaR itself isn't the noisy
   bottleneck (20k paths → ~1000 in the CVaR-95 tail).
2. **Shifted test CVaR-95**: same model, evaluated on a second fixed test set of equal
   size (`n_test = 20_000`), drawn from the same market with the shifted leverage
   params from §1 (`beta=-0.45, kappa_L=0.75`) — a single fixed offset, not a grid.

Both test sets are shared across every method/hyperparam/n combination (see §5) so
comparisons are paired, not just averaged.

## 5. Dataset generation (once, upfront)

Simulation (LSV, fine sub-stepped, `n_time=100`) is the expensive part — generate all
paths once per seed and cache to disk, then have every method/hyperparam config read
from the cache instead of resimulating.

Per seed:
- **Train pool**: simulate `n_max = 1000` paths once; for each smaller `n` in the grid,
  use the **first `n` paths of the pool** (nested prefixes) rather than resampling —
  standard trick to keep the `n`-scaling curve smooth (a run at `n=500` and one at
  `n=1000` share their first 500 paths) and to avoid `5 × n_max` worth of simulation.
  k-fold splits are computed on top of whichever prefix is in use for that `n`.
- **Test set** (in-dist): simulate `n_test = 20_000` paths independently of the train
  pool, in-dist leverage params.
- **Shifted test set**: simulate `20_000` paths, shifted leverage params (§1/§4).

Cache format: raw simulated tensors (tradable prices, features, variance) keyed by
`(seed, leverage_variant, split, index-range)`, e.g. one file per
`(seed, "train_pool")`, `(seed, "test")`, `(seed, "test_shifted")`. Everything
downstream (training, k-fold slicing, evaluation) just indexes into these arrays — no
re-simulation anywhere else in the pipeline.

## 6. Seeds

3 seeds (trimmed from 5 for compute, see §7). Each seed gets its own
independently-simulated train pool + both test sets (§5); results reported as
mean ± spread across seeds, paired within seed across methods/`n`.

## 7. Compute budget

Per `(n, seed)`, with the full 3×3×3 grids (`cvar_of_average`/`relaxed_kl_wdro`) and
`k=5`: hyperparam configs needing 5-fold CV = `9 + 27 + 27 = 63` → `63 × 5 = 315`
fold-trainings, plus `3` full-`n` refits = `318` trainings; `× 5 n-values × 5 seeds
≈ 8,000` total. Too many — applying mitigations up front:

1. **Trim `cvar_of_average`/`relaxed_kl_wdro` grids to 2 values/axis** (`cvar` is
   already small, left as-is): `2×2×2 = 8` configs each instead of 27. New
   per-`(n,seed)` config count: `9 + 8 + 8 = 25`.
2. **Seeds: 3 instead of 5** (§6).
3. **`k = 3` instead of 5** (§3): `25 × 3 = 75` fold-trainings + `3` refits = `78`
   trainings per `(n, seed)`.

Total: `78 × 5 n-values × 3 seeds ≈ 1,200` training runs — down from the original
~8,000 estimate (dropping `big_batch`, §2, and starting `n` at 100 instead of 50
further reduced the count on top of the two compute mitigations). Simulation cost
(`n_substeps=10`-fine LSV) is paid once via the §5 cache, not per training run, and is
unaffected by these mitigations.

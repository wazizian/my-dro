# Current Situation

The codebase already has WDRO wired into the training loop.

- `deep_hedging.py` trains the hedge strategy and the WDRO dual state together.
- `wdro.py` defines the dual variables and turns the unconstrained `raw_lambda` into a nonnegative `lambda` with `softplus`.
- Both WDRO objectives are present:
  - `entropic`
  - `joint_cvar`
- The `joint_cvar` branch currently uses a single global `eta`, not one threshold per base trajectory.
- The proposal mechanism already supports `k = 0, 1, 2, ...` ULA steps through `robust.ula_steps`.

So yes: `lambda` is being optimized during WDRO training, not held fixed.

## Update (2026-07-03): k=1 tuning done, all three modes pass

Ran the small learning benchmark (`learning_benchmark.py`) to completion for steps 1-3 below.

- `k = 0` (base Gaussian proposal) already passed for both `entropic` (`kl_epsilon=0.03`, 1000 steps) and `joint_cvar` (`kl_epsilon=0.1`, 1500 steps): see `benchmark/learning_smoke_entropic_eps_003` and `benchmark/learning_smoke_joint_1500`.
- `k = 1` (one ULA refinement step) originally failed for both objectives at default `proposal_scale=0.05`, `ula_step_size=0.0025`:
  - `entropic`: risk not improving (validation CVaR got worse).
  - `joint_cvar`: not tail-calibrated — active-tail fraction collapsed to ~0.8-1.6% (target 2-15%), because the tighter proposal caused `eta` to overshoot relative to the true tail quantile.
- Fix found by sweeping `proposal_scale` x `ula_step_size` (300-step probes, then 1500-step confirmation): **`proposal_scale=0.1`, `ula_step_size=0.0025`** fixes both. For `joint_cvar` this alone still left active-tail borderline-low over long training (drifts down as the hedge improves), so `alpha` was also lowered from `0.95` to `0.90` to give the tail-quantile calibration more margin; `alpha=0.90` gives active-tail ≈2.6% (`alpha=0.92` also passes but sits right at the 2% floor, `alpha=0.95` no longer used at k=1).
  - Confirmed passing runs: `benchmark/learning_smoke_entropic_k1_tuned`, `benchmark/learning_smoke_joint_k1_tuned` (both `ROBUST_PASS=True`).
- Added `alpha` as a proper CLI field on `LearningBenchmarkConfig` in `learning_benchmark.py` (it was previously hardcoded to `0.95` in `experiment_config`), so this alpha tuning is reproducible from the command line instead of only via a monkeypatched probe script.
- Step 3 (compare both WDRO objectives vs. nominal CVaR baseline at a matched 1500-step budget) is done: `benchmark/comparison_tuned_k1/summary.csv`. All three modes pass (`learned=True`). As expected for DRO, `nominal` has the best empirical validation CVaR-95 improvement (+60.7%) since it directly optimizes that metric; `entropic` trades CVaR-95 away (-10.6%, though its own dual objective improves ~49x and stop-loss still improves) and `joint_cvar` lands in between (+34.1%) since its dual objective is CVaR-shaped. This tradeoff is expected, not a bug — DRO sacrifices average/CVaR performance for robustness to distribution shift.

## Next Steps

1. ~~Tune the entropic WDRO hyperparameters first.~~ Done — see update above.
2. ~~Tune the joint-CVaR WDRO with `k = 1`.~~ Done — see update above (`proposal_scale=0.1`, `ula_step_size=0.0025`, `alpha=0.90`).
3. ~~Compare the two WDRO objectives against the nominal CVaR baseline using the same small benchmark budget.~~ Done — see `benchmark/comparison_tuned_k1/summary.csv`.
4. Only after the small benchmark is stable, move to a broader hyperparameter sweep. **Remaining.**

   The 1500-step budget above is still short (`batch_size=64`, small validation/eval sizes). The goal now is to confirm the `k=1`, `proposal_scale=0.1`, `ula_step_size=0.0025` region holds up under:

   - More training steps / larger batch size (current runs are a smoke test, not a converged fit).
   - `k = 2` ULA steps, per `robust.ula_steps` support already in the proposal mechanism.
   - A real grid around `rho` (currently untouched at the `0.075` default) and `proposal_samples` (untouched at `8`), which the original plan called out but this pass did not touch.
   - Multiple seeds, since all runs above used `seed=0` only.

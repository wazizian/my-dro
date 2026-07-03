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

What remains is tuning.

## Next Steps

1. Tune the entropic WDRO hyperparameters first.

   The main knobs are:

   - `rho`
   - `kl_epsilon`
   - `tau`
   - `proposal_samples`
   - `proposal_scale`
   - `ula_steps`
   - `ula_step_size`

   Start with a small grid around the current defaults and keep the strategy architecture fixed.

2. Then tune the joint-CVaR WDRO with `k = 1`.

   This is the first proposal setting that actually uses one ULA refinement step instead of the base Gaussian proposal.
   The key things to retune are:

   - `rho`
   - `alpha`
   - `proposal_samples`
   - `proposal_scale`
   - `ula_step_size`

   For `k = 1`, `proposal_scale` and `ula_step_size` should be treated together, since they control the initial spread and the refinement scale of the proposal.

3. Compare the two WDRO objectives against the nominal CVaR baseline using the same small benchmark budget.

   Keep the benchmark short enough to detect learning and proposal health before running larger sweeps.

4. Only after the small benchmark is stable, move to a broader hyperparameter sweep.

   The goal is to avoid spending time tuning a configuration that is not learning at all.

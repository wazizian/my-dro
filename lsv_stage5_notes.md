# Stage 5 — relaxed convex-order penalty: implicit design choices

Spec: plan `steady-swinging-lagoon.md` Part B, decisions locked via AskUserQuestion (stock leg only,
explicit `convex_order_strikes` config field). Implementation: `wdro.py` (`RobustConfig` fields,
`RobustState.raw_lambda_ij`, `convex_order_penalty` helper), `deep_hedging.py`
(`robust_hedging_loss` splice, `train_step` gradient-zeroing).

1. **`raw_lambda_ij` given a `default_factory` (`jnp.zeros((1,1))`) on `RobustState` rather than
   editing every existing test/call site.** Plan said "always present in the pytree ... inert when
   unused", but didn't anticipate that adding a new *required* field would break ~13 existing
   `RobustState(...)` call sites across `test_wdro.py`/`test_deep_hedging.py`/`learning_benchmark.py`
   that construct it with only 4 positional args. Rather than touch every one of those (larger,
   noisier diff, and risks subtly changing tests unrelated to this feature), gave the new field a
   default. The default shape `(1,1)` is arbitrary/never semantically meaningful — it's only ever
   read when `robust_config.enforce_convex_order=True`, and none of those existing call sites set
   that flag, so the mismatch between the default's shape and whatever `(n_strikes, n_time)` a real
   run would need is harmless. `deep_hedging.py`'s own real construction (in `train()`) always
   explicitly passes a correctly-shaped `raw_lambda_ij`, never relying on the default.

2. **Penalty computed on `proposed_prices[..., 0]` (index 0 = stock leg)**, per the user's locked
   decision (stock leg only, not variance swap). This relies on `tradables = jnp.stack((stock,
   variance_swap), axis=-1)`'s fixed ordering in `simulate_heston`/`simulate_lsv`/`decode_trajectory`
   (stock always index 0) — true for both Heston and LSV markets, not re-verified for
   `black_scholes` (single-asset market, `n_assets=1`, so `[..., 0]` is still correct there too,
   trivially).

3. **Penalty added into `pathwise_losses` (i.e. into `f`) before every dual dispatch, not into any
   individual `*_dual` function** — exactly as planned, a single splice point. This means the penalty
   participates identically in whichever objective is selected (`entropic`, `joint_cvar`,
   `per_sample_cvar`, `relaxed_kl_cvar`, `plain_average`, `plain_cvar_of_average`) without needing
   per-dual code. One consequence not explicitly spelled out in the plan: `plain_average_dual`/
   `plain_cvar_of_average_dual` (the confound-isolation ablation baselines, which by design ignore
   `transport_cost`/`lambda_`) *do* still pick up the convex-order penalty if `enforce_convex_order`
   is set, since it's folded into `pathwise_losses` upstream of the dispatch — i.e. the penalty isn't
   gated to "only WDRO-style objectives", it applies to any objective if the flag is on. This seems
   correct/intended (the penalty is about proposal-path realism, not about the WDRO dual structure
   specifically) but is worth flagging since those two duals' docstrings say they intentionally have
   "no worst-case tilt, no transport-cost penalty" — the convex-order penalty is a third, independent
   mechanism, not covered by that description.

4. **`log_target`'s single-sample ULA/importance-weight path also gets the penalty** (not just the
   main batched path), per the plan's explicit instruction to steer proposal generation itself, not
   just penalize post-hoc. Implemented as a second call to the same `convex_order_penalty` helper
   with `n=1`. Given the default `ula_steps=0` for the sweep's `relaxed_kl_wdro` config (per
   `lsv.md` §2, "the config the prior report found best"), `log_target`'s *gradient* w.r.t. sample
   position is never actually used in that default configuration (no ULA refinement steps to take
   gradients over) — but `log_target` is still called to build importance/reweighting log-densities
   in some code paths, so keeping the penalty there is defensive/complete rather than dead code.
   Not verified via a dedicated `ula_steps>0` smoke test (out of scope: `lsv.md`'s own config never
   uses `ula_steps>0`) — flagging as an untested code path if `enforce_convex_order=True` is ever
   combined with `ula_steps>0`.

5. **`fix_convex_order_lambda` gradient-zeroing condition is `not enforce_convex_order OR
   fix_convex_order_lambda`** (i.e. always zero the gradient when the feature is off, in addition to
   when explicitly fixed) — this is slightly broader than the plan's literal wording ("mirrors the
   existing `fix_lambda` flag"), which only described the `fix_convex_order_lambda=True` case
   explicitly. Added the `not enforce_convex_order` half defensively: since `raw_lambda_ij` is never
   read in the loss when the feature is off, its gradient would be structurally zero anyway (no
   computational path from it to the loss) — this branch is redundant with that fact, not a behavior
   change, just an explicit belt-and-suspenders zeroing matching the existing `location`/`scale`
   pattern (always zeroed regardless of relevance).

6. **`convex_order_lambda_init` reuses the exact same softplus-inverse pattern as `initial_lambda`**
   (`raw = log(expm1(init))`), broadcast to the full `(n_strikes, n_time)` shape via `jnp.full` —
   every strike/time-pair multiplier starts at the same fixed initial value, not strike- or
   time-varying. No spec/user guidance suggested otherwise; simplest choice consistent with the
   existing single-scalar `initial_lambda` convention.

7. **Independent review (no bugs found)**: verified the `convex_order_penalty` broadcast/transpose
   analytically (correct sign/index correspondence), confirmed both splice points use the stock leg
   with correct dtype casting, confirmed `enforce_convex_order=False` traces zero extra ops (true
   no-op, not a zero-valued tensor), and ran a live smoke test (Heston, n=25, n_time=6, 40 steps):
   `fix_convex_order_lambda=False` moved `softplus(raw_lambda_ij)` from all-1.0 to
   `[0.918,1.071,1.014,0.963,0.925,0.933]` (gradient flows); `=True` stayed exactly all-1.0
   (gradient correctly zeroed). Full suite 47/47 passing throughout. One accepted risk confirmed
   live (not just theoretical): item #3's claim that `plain_average`/`plain_cvar_of_average` also
   pick up the penalty when the flag is set — traced and confirmed true, added a docstring caveat
   to both (`wdro.py`) rather than changing behavior, since the plan's single-splice-point design is
   intentional and correct. `RobustState.raw_lambda_ij`'s shape (built from `config.market.n_time`
   at construction) vs. `robust_hedging_loss`'s own `n_time` (derived from the `prices` argument's
   shape) is a latent, structurally-unenforced invariant — no live mismatch found in any current
   call site, flagged as a footgun for future callers rather than fixed now (out of scope).

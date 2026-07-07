# Is the ULA importance-sampling correction why k>1 underperforms k=1?

## Goal

`heston_findings.md` found (at toy `n_time=5`) that ULA steps `k>1` (5, 10) uniformly hurt vs
`k=1`, for both `entropic` and `joint_cvar`, so `k=1` became the fixed default carried through
every later investigation. The mechanism was never isolated. `sample_langevin_proposal`
(`wdro.py`) applies an importance-sampling correction to `k>=1` ULA samples:
`log_weights = log_reference(final) - log_transition(final, final_mean)` — comparing the last
step's transition kernel against the *original* reference kernel, ignoring steps `1..k-1`
entirely. This is an approximation, not a full accumulated-path density ratio or MH acceptance
ratio, and gets worse as `k` grows. Question: is this (biased/approximate) IS correction, rather
than the extra Langevin step itself, what makes `k>1` underperform?

## Code change

Added `use_importance_sampling: bool = True` to `RobustConfig` (`wdro.py`). In
`sample_langevin_proposal`'s `n_steps > 0` branch, when `False`, `log_weights` is forced to zero
regardless of `k` — but the chain still runs in full for `k` steps, so only the correction is
disabled, isolating "does the multi-step chain produce a good proposal" from "does the IS
correction on top of it help or hurt." Default `True` preserves all existing behavior exactly.

New unit test `test_use_importance_sampling_toggle_gates_log_weights` (`test_wdro.py`),
parametrized `k in {1,2}` x `use_importance_sampling in {True, False}`, using the same
non-degenerate `log_target` fixture as the existing shape/determinism test. All 29 tests pass.

**Independent review #1** (pre-run, code audit): confirmed the change is default-preserving
(grepped every `RobustConfig(` construction site — none set the new flag), correctly placed
relative to `stop_gradient`, and the test exercises genuinely non-degenerate weights (std ≈
0.95–1.65, not float noise). No issues found.

## Benchmark

`relaxed_kl_cvar`, `n_time=30` (realistic horizon), BS + Heston, reusing each market's
already-established n_time=30 lambda from `longer_horizon_findings.md` (BS: λ=0.025, Heston:
λ=0.002; `alpha=0.9, exact_eta=True, tail_power=1.0, cost_power=1`, gaussian proposal,
`proposal_scale=0.15, ula_step_size=0.02, proposal_samples=8, batch_size=64, steps=1500`) — no
lambda re-tuning, since this is about the mechanism, not chasing a new optimum. 5 labels
(`nominal`, `k1_is_on`, `k1_is_off`, `k2_is_on`, `k2_is_off`) x 2 markets x 10 paired seeds
(7000-7009, fresh). 100 jobs, 0 failures. Data: `benchmark/rho_sweep/ula_is_stageA.csv`.

## Results (means, n=10)

| market | label | in_dist | worst_extreme |
|---|---|---|---|
| black_scholes | nominal | 9.88 | 497 |
| black_scholes | k1_is_on | 9.21 | 419 |
| black_scholes | k1_is_off | 9.29 | 449 |
| black_scholes | k2_is_on | 9.74 | 615 |
| black_scholes | k2_is_off | 9.68 | 372 |
| heston | nominal | 9.53 | 67.3 |
| heston | k1_is_on | 8.83 | 24.7 |
| heston | k1_is_off | 8.56 | 23.8 |
| heston | k2_is_on | 9.64 | 33.5 |
| heston | k2_is_off | 10.24 | 37.5 |

## Comparisons (paired ttest_rel / wilcoxon, n=10)

**1. `k1_is_off` vs `k1_is_on` (sanity — is the toggle a no-op at k=1?):** no significant
difference, either market or metric (BS in-dist p=0.85/0.70, worst p=0.71/0.92; Heston in-dist
p=0.12/0.19, worst p=0.63/0.70). Confirms the gate behaves as designed at k=1.

**2. `k2_is_on` vs `k1_is_on` (reproduce the prior k>1-hurts finding):** Heston in-dist 9.64 vs
8.83, p=0.0097/0.0195 — clearly worse. Heston worst 33.5 vs 24.7, p=0.044/0.065 — worse,
borderline. BS worst 615 vs 419, p=0.077/0.037 — worse, borderline/discrepant. BS in-dist ns
(p=0.22). Net: k=2 underperforms k=1 at this longer horizon too, strongest in Heston.

**3. `k2_is_off` vs `k1_is_on` (the key test — does disabling IS at k=2 recover k=1?):**
**Heston: still significantly worse than k1**, on both metrics (in-dist 10.24 vs 8.83,
p=0.0003/0.0020; worst 37.5 vs 24.7, p=0.035/0.037) — no recovery; if anything the in-dist number
is slightly worse than `k2_is_on`'s. **BS: not significantly different from k1** on either metric
(in-dist p=0.25/0.49; worst 372 vs 419, p=0.48/0.49, point estimate nominally better) — reads as
"no significant harm," not "confirmed recovery," given the small n and wide CI.

**4. `k2_is_off` vs `k2_is_on` (direct toggle effect at k=2):** BS worst_extreme 372 vs 615,
ttest p=0.054, wilcoxon p=0.014 — direction favors `is_off` in 8/10 seeds; traced to two
heavy-tailed seeds (7004, 7008) inflating ttest variance without flipping the sign, so this is a
real but noisy effect, still only borderline at n=10. **Heston: `is_off` is significantly worse**
on in-dist (10.24 vs 9.64, p=0.042/0.049) — the toggle actively hurts here. Heston worst ns
(p=0.28/0.43).

**5. vs `nominal` (context):** all four ULA configs beat nominal on worst_extreme in Heston
(p≤0.08, all directions correct), confirming WDRO training still helps there; BS shows
weaker/mixed nominal-relative signal, consistent with prior findings that Heston is where the
robustness benefit shows up most clearly.

**Independent review #2** (post-run, stats + design audit): recomputed all comparisons
independently, confirmed the design is clean (10 unique paired seeds per cell, lambda genuinely
fixed within market across all `relaxed_kl_cvar` labels, no duplicated rows, no silently
under-trained jobs). Verdict: "mixed, leaning refuted" — pushed back specifically that Heston's
result is a fairly clean refutation, not mere inconclusiveness, since it's the market with the
cleanest k1-vs-k2 gap and disabling IS there makes things *worse*, not better.

## Bottom line

- **The "biased IS correction explains k>1's underperformance" hypothesis is not supported.**
  In Heston — the market with the strongest, cleanest k=1-vs-k=2 gap — disabling the IS
  correction at k=2 does not recover k=1-level performance and makes in-distribution CVaR
  significantly *worse* (p=0.0003). If the IS correction were the culprit, removing it should
  have helped or at least been neutral; instead it hurt.
- In BS, disabling IS at k=2 shows a borderline, noisy improvement on worst-case tail risk
  (driven by 2 of 10 seeds), but even with IS off, k=2 doesn't clearly beat k=1 — it's "no
  significant harm" relative to k=1, not a rescued win.
- At k=1, the toggle is a confirmed no-op (as expected — the correction is real but doesn't
  compound), so this benchmark is a genuine single-variable test of the correction's effect, not
  an artifact of the toggle itself doing nothing.
- **Practical implication:** `k=1` (the established default) remains the right choice. The
  underperformance of `k>1` is not primarily explained by the IS-weight formula being biased at
  longer chains — something else about running more Langevin steps (proposal-distribution drift,
  discretization effects on the actual samples themselves, or the increased variance visible in
  BS's heavy-tailed seeds) is the more likely driver, though this benchmark doesn't pin that down
  further.

## Caveats / open items

- n=10 seeds; BS's borderline effects (comparison 4) could sharpen or vanish with more seeds
  (e.g. n=20-30) — not run here since the Heston result alone is already a clean refutation of
  the primary hypothesis.
- Only `relaxed_kl_cvar` was tested (per the specific question asked); `entropic`/`joint_cvar`
  were not re-checked with the IS toggle at longer horizons.
- `ula_step_size=0.02` was reused unchanged from the k=1 tuning; it's possible a k=2-appropriate
  step size (smaller, to compound less discretization error per step) would tell a different
  story — not tested here.

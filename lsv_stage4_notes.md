# Stage 4 — risk-neutral Heston recalibration + 3-tier shift ladder: implicit design choices

Spec: plan `steady-swinging-lagoon.md` Part A, decisions locked via AskUserQuestion (kappa=3.0,
vol_of_variance=0.4; weak/medium/hard = 1.5x/2x/3x). Implementation: `deep_hedging.py`
`MarketConfig` defaults, `lsv_sweep.py` `LEVERAGE_PARAMS`/`run_one`/`CSV_FIELDNAMES`, `lsv.md` §1/§4.

1. **`theta`/`rho`/`v0` (long_variance, correlation, initial_variance) left unchanged.** Plan
   explicitly said these were "already literature-typical" per the web-search grounding (rho=-0.7
   directly confirmed, theta=0.04↔20% vol and v0=0.04 matching the same). Only `kappa`/
   `vol_of_variance` changed. No independent re-verification beyond the search done during planning.

2. **`MarketConfig` defaults changed globally, not just for the LSV study.** `kappa`/
   `vol_of_variance` are shared fields also used by plain `simulate_heston` (non-LSV Heston
   benchmark) and by `black_scholes`-unrelated code paths that read `MarketConfig` defaults (e.g.
   any existing test or script that constructs `MarketConfig()` without overriding `kappa`/
   `vol_of_variance`). This silently changes behavior for anything relying on the old defaults
   outside the LSV study. Not scoped to check every call site for whether it depends on the old
   `kappa=1.0`/`vol_of_variance=2.0` values — flagging as a blast-radius risk for review, since the
   plan said "pure default-value change" but didn't audit non-LSV callers.

3. **Shift-tier naming**: renamed `LEVERAGE_PARAMS["shifted"]` to `LEVERAGE_PARAMS["weak"]` (same
   values, `beta=-0.45, kappa_L=0.75`) rather than keeping `"shifted"` as an alias for `"weak"`.
   This is a **breaking rename** — any code/notebook/analysis outside `lsv_sweep.py` that referred
   to the `"shifted"` key or the CSV's `test_cvar95_shifted` column (e.g. the already-committed
   `lsv_sweep_results.csv` from the prior full run, and `lsv_report.typ`) will no longer match the
   new schema. The prior sweep's results remain valid under the *old* schema/params — they are not
   retroactively invalidated by this rename, but any script hardcoding the old column name breaks.
   Chose a clean rename over keeping a `"shifted"` alias since the plan's CSV schema explicitly
   listed `test_cvar95_weak/medium/hard` with no `test_cvar95_shifted` survivor.

4. **`_TAG_OFFSETS` extended with new integer offsets (3, 4) for `test_medium`/`test_hard`**,
   keeping `train_pool=0`, `test_indist=1`, and reassigning `test_shifted=2` to `test_weak=2` (same
   integer, same cache-key derivation, so a `weak`-tier dataset simulated under the *old* Heston
   core params would collide on cache key with one simulated under the *new* params if pointed at
   the same `--cache-dir`). This is the same class of hazard stage 3's cache-collision fix
   addressed (there, `n_paths` wasn't in the filename; here, the Heston *params* aren't in the
   filename at all, only `(seed, tag, n_paths)`). Not fixing this now since the plan didn't call for
   a cache-key scheme change and the existing `lsv_cache/` from the prior run should simply be
   deleted/regenerated before any new sweep runs under the new params — flagging prominently so
   review/smoke-test doesn't silently load stale-param cached data.

5. **`SHIFT_TIERS = ("weak", "medium", "hard")` fixed tuple, ordering matters for CSV row
   construction** (`row[f"test_cvar95_{tier}"]` in a `for tier in SHIFT_TIERS` loop) — order is
   cosmetic only (dict-keyed access, not positional), no functional risk, noting for completeness.

6. **No change made to `CANONICAL_ALPHA`, `K_FOLDS`, `STEPS`, `N_GRID`, `N_TEST`, `N_MAX`, or any
   compute-budget constant.** Plan explicitly scoped this stage to params/tiers only; compute-cost
   re-estimation (now 2 extra test-set evaluations per cell: 4 test sets vs 2) is deferred to the
   plan's "separate follow-up" step, not addressed here.

7. **Fix applied after review**: item #4's cache-collision risk was not hypothetical — the
   pre-existing `lsv_cache/` directory (237MB, untracked) still held `train_pool`/`test_indist`
   datasets simulated under the OLD Heston params (kappa=1.0, vol_of_variance=2.0), and would have
   been silently reused by any new run pointed at the same `--cache-dir` (cache keys don't encode
   market params, only `(seed, tag, n_paths)`). Deleted `lsv_cache/` entirely before smoke testing.
   No cache-key scheme change made (out of scope for this stage) — anyone regenerating the sweep
   must ensure `lsv_cache/` doesn't contain pre-recalibration data before pointing a run at it.

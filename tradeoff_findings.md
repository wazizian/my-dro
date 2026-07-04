# WDRO hyperparameter hunt: does a genuine in-distribution/robustness tradeoff appear?

Investigation date: 2026-07-03. Goal: tune every available knob for both `entropic` and
`joint_cvar` WDRO objectives until a clear tradeoff (worse in-distribution CVaR95, better
worst-case-stress CVaR95, relative to nominal CVaR training) appears for both. ~150
training runs across BS and Heston markets. All raw CSVs are in `benchmark/rho_sweep/`.

## TL;DR

- **entropic**: a genuine tradeoff was found. At `initial_lambda=0.1` (fixed, not
  dual-optimized), BS: in-dist CVaR95 goes from nominal's 1.79 to 2.62 (+46%), while
  worst-case CVaR95 under a vol=1.5 shock drops from nominal's ~47-50 to 30.2 (-35%).
  Confirmed weaker on Heston too.
- **joint_cvar**: no tradeoff was ever found (rho up to 300, fixed lambda, local knobs all
  swept) *until* generalizing the CVaR (L¹ tail norm) to an Lᵖ tail norm with p≥2, combined
  with computing `eta` exactly per training batch instead of dual-ascending it. That
  combination doesn't produce a tradeoff either — it beats nominal outright on **both**
  in-dist and worst-case (in-dist ties nominal, worst-case ~10-15% better) — but the margin
  is only about one nominal-seed standard deviation, so treat this as a promising, not yet
  fully confirmed, lead (see "Open questions" below).
- **per_sample_cvar** (new objective, per-anchor exact `eta` instead of one shared batch
  `eta`) failed outright — with only 8 proposals per anchor, the per-anchor 90th-percentile
  estimate is dominated by order-statistic noise (in-dist 4-5, worse than nominal on both
  axes at every setting tried).
- **`ula_temper`** (forcing joint_cvar/per_sample_cvar's ULA target to use entropic's
  temperature-scaled gradient) does not explain entropic's edge — neutral-to-harmful when
  tried on joint_cvar/per_sample_cvar.

## 1. Knobs tried, in order

| Knob | Where | What it does | Verdict |
|---|---|---|---|
| `rho` | dual-optimized lambda | transport radius | No effect — dual-ascended lambda ignores it, converges to a fixed value (~2.2 entropic, ~0.4 joint_cvar) regardless of rho |
| `fix_lambda` + `initial_lambda` | `wdro.py` `RobustConfig` | hold lambda fixed instead of dual-ascending | **The key unlock for entropic.** Clean unimodal curve in λ; sweet spot ≈0.05-0.1 (BS), ≈0.005-0.02 (Heston). Tried for joint_cvar too — no help by itself. |
| `proposal_scale` (σ in π₀) | `RobustConfig.proposal_scale` | spread of the ULA/reference proposal | Real sweet spot ≈0.05-0.2 depending on config; too small under-explores, too large wastes capacity on unrealistic scenarios |
| `cost_power` | `RobustConfig.cost_power` (new) | transport cost: 1 = ‖·‖ norm, 2 = ½‖·‖² (default) | norm cost (1) helped entropic's Heston worst-case specifically |
| `proposal_family` | `RobustConfig.proposal_family` (new) | π₀: gaussian vs laplace | **Correction (see §7): laplace actually won the entropic 24-combo BS sweep** (cost_power=2, scale=0.2), and a targeted λ sweep at that point beat gaussian's best entropic point outright. No benefit for joint_cvar's Lᵖ/exact-η recipe, where gaussian stays best. |
| `ula_step_size` | `RobustConfig.ula_step_size` | Langevin step size | Unimodal sweet spot ≈0.0025-0.02; too large diverges (blew up at 0.04) |
| `tail_power` (p) | `RobustConfig.tail_power` (new) | joint_cvar/per_sample_cvar: Lᵖ norm of tail excess instead of L¹ (CVaR) | **p≥2 is where joint_cvar finally started working** — see §3 |
| `exact_eta` | `RobustConfig.exact_eta` (new) | joint_cvar: batch-exact (1-α) quantile (stop-gradient) instead of dual-ascended η | Alone (p=1) insufficient; combined with p≥2, opens the win |
| `per_sample_cvar` objective | `RobustConfig.objective` (new) | per-anchor exact η (conditional constraint, not joint) | Failed — 8 samples/anchor too few for a stable 90th-percentile estimate |
| `ula_temper` | `RobustConfig.ula_temper` (new) | apply entropic's temperature-scaling to the ULA target for other objectives too | Neutral-to-harmful; not the explanation for entropic's edge |

All new `RobustConfig` fields are implemented in `wdro.py` and wired through
`deep_hedging.py` (`robust_hedging_loss`, `make_robust_steps`). All 30 existing unit tests
(`test_wdro.py`, `test_deep_hedging.py`) still pass.

## 2. entropic: the tradeoff, in numbers

Stress grid: BS vol ∈ {0.10, 0.20, 0.30, 0.60, 0.90, 1.20, 1.50}; "worst_extreme" = worst
CVaR95 across all of these except the in-distribution point (vol=0.20).

| config | in_dist | worst_extreme |
|---|---|---|
| nominal (mean of 3 seeds) | 1.77 | 47.0 (σ≈2.5 across seeds) |
| entropic λ=0.03 | 2.71 | 34.8 |
| entropic λ=0.05 | 2.58 | 31.9 |
| **entropic λ=0.1** | **2.62** | **30.2** |
| entropic λ=0.2 | 3.05 | 51.1 |
| entropic λ=1.0 | 4.23 | 69.6 |
| entropic λ=32 | 6.25 | 80.3 |

(local knobs: `proposal_scale=0.2`, `cost_power=1`, `ula_step_size=0.02`, k=1,
8 proposals/anchor)

Heston (worst_extreme includes a "ultra" scenario: long_variance=0.25, vol_of_var=8,
correlation=-0.999, kappa=0.1):

| config | in_dist | worst_extreme |
|---|---|---|
| nominal | 2.15 | 4.47 |
| entropic λ=0.005 | 3.22 | **2.87** |
| entropic λ=0.02 | 2.70 | 3.66 |

This is a real tradeoff: at moderate stress severity (vol 0.15-0.30) entropic never beats
nominal, but once the stress is pushed far enough out (vol≥0.9, or Heston's "ultra"
scenario) nominal's hedge breaks down and paying an in-distribution premium buys a real,
substantial reduction in tail loss.

## 3. joint_cvar: from "never wins" to "wins on both axes" via Lᵖ + exact η

**Everything tried before Lᵖ/exact-η failed.** rho up to 300, `fix_lambda` from 0.005 to 1.0,
`proposal_scale` from 0.02 to 0.3, `cost_power` 1 vs 2 — joint_cvar's worst_extreme was
*always* worse than nominal's (see `tradeoff_hunt.csv`, `joint_cvar_fix_lambda_hunt.csv`).
Its dual-ascended λ also ignores rho, converging to ~0.4 regardless (10x rho range tested)
— the identical pathology found earlier for entropic's λ.

Generalizing the CVaR-shaped dual (`joint_cvar_dual` in `wdro.py`) to an Lᵖ norm of the tail
excess (per `note_wdro_solve.typ`'s joint-constraint dual with general Hölder conjugate p),
and switching η from a dual-ascended variable to the exact per-batch (1-α)-quantile
(stop-gradient; batch×proposals=512 samples per step, so the bias is small), changed the
picture completely:

| config (λ fixed=0.1, `proposal_scale=0.15`, `cost_power=1`) | in_dist | worst_extreme |
|---|---|---|
| nominal (mean of 3 seeds) | 1.77 | 47.0 |
| p=1, exact_eta=False (old CVaR, dual-ascended η) | 2.42 | 52.3 |
| p=1, exact_eta=True | 1.88 | 64.3 |
| p=1.5, exact_eta=True | 1.78 | 55.6 |
| **p=2.0, exact_eta=True** | **1.75** | **42.4** |
| **p=3.0, exact_eta=True** | **1.76** | **41.3** |

At p=2-3 with exact η, joint_cvar matches nominal's in-distribution CVaR95 and beats its
worst-case CVaR95 by ~10-15%. This isn't the tradeoff shape that was being hunted for — it's
an outright (mild) improvement on both axes. Mechanistically, raising p makes the dual
penalize the single worst excess in a batch more sharply than the L¹/CVaR average did,
which seems to act like extra regularization against tail overfitting, similar in spirit to
what fixed small λ did for entropic.

**Caveat (from a follow-up user request to control for noise):** nominal's own worst_extreme
varies by about ±2.5-3 across 3 seeds (44.1, 46.9, 50.2), so the joint_cvar win (41.3-42.4) is
only around one nominal-seed standard deviation below the nominal mean. Directionally
consistent (both p=2 and p=3 beat all three nominal seeds), but this should be treated as a
promising lead, not a fully confirmed result, until multi-seed runs of the winning joint_cvar
config are done.

## 4. Negative results (from the user's explicit requests)

- **`per_sample_cvar`** (conditional-Lq dual, `@conditional-dual` in the note — one η per
  training anchor, not one shared η per batch): implemented in `wdro.py` as
  `per_sample_cvar_dual`. With only `proposal_samples=8` per anchor, the per-anchor
  (1-α)=0.9-quantile estimate is essentially just picking near the max of 8 samples — very
  high variance. Result: in-dist CVaR95 of 4.0-5.4 (nominal 1.77) and worst_extreme 52-134
  (nominal ~47) at every (p, λ) combination tried. The user's hypothesis that "even 8 should
  be fine" for the quantile bias did not hold in practice — bias may be small but variance is
  large at this sample count, and that variance directly perturbs the tail-shaping term every
  step.
- **`ula_temper`** (apply entropic's `temperature = kl_epsilon + λ·τ` scaling to the ULA
  target for joint_cvar/per_sample_cvar too, to test whether that's why entropic's local
  exploration works better): roughly neutral for joint_cvar p=2 (41.5 vs 42.4 worst_extreme),
  clearly harmful for joint_cvar p=3 (54.1 vs 41.3) and per_sample_cvar p=1 (175 vs 104).
  Temperature-scaling the ULA gradient is not the explanation for entropic's edge.

## 5. Interpretation

Entropic's softmax/exp-tilted reweighting of proposals generalizes to genuinely unseen,
far-out-of-distribution shocks in a way that the original CVaR-thresholded (L¹) local
reweighting did not — likely because softmax reweighting has support everywhere (soft,
smoothly graded), while a hard `(x - η)_+` threshold with L¹ averaging only reacts to
samples that already cross η, and with a dual-ascended η that pathology compounds (η, like
λ, was found not to respond meaningfully to configuration). Raising the tail norm's exponent
p sharpens joint_cvar's reaction to the single worst sample in a batch rather than the tail
average, which is closer in spirit to entropic's soft-max weighting (softmax is essentially
an L^∞-like limit as sharpness increases) — that may be why p≥2 closes most of the gap.

## 7. Correction: laplace π₀ actually helps entropic, not joint_cvar

§1 originally claimed "gaussian always won" the 24-combo BS knob sweep (`knob_sweep_bs.csv`).
Re-checking that data on request: **that was wrong.** The best row in the whole sweep was
entropic with `cost_power=2, proposal_family=laplace, proposal_scale=0.2`: in_dist=2.53,
worst(vol=0.30)=3.60 — beating what had been reported as the best gaussian point (cost_power=1,
scale=0.2: 2.83/3.97) on both axes.

Follow-up: swept fixed λ at that laplace point (`cost_power=2`, `scale=0.2`) under the harsh
grid (vol up to 1.5):

| config | in_dist | worst_extreme |
|---|---|---|
| nominal | 1.77 | 47.0 |
| gaussian entropic λ=0.1 (previous best) | 2.62 | 30.2 |
| **laplace entropic λ=0.01** | **2.53** | **27.1** |
| laplace entropic λ=0.02 | 2.48 | 31.7 |
| laplace entropic λ=0.05 | 2.77 | 43.9 |

Laplace λ=0.01 beats the gaussian frontier point on **both** in-dist and worst-extreme —
a genuine improvement, not just a repositioning along the tradeoff curve.

On Heston, laplace does not overtake gaussian: best laplace point (λ=0.02: in_dist=3.63,
worst=3.36) is behind gaussian's best there (λ=0.005: in_dist=3.22, worst=2.87), though
laplace still beats nominal (2.15/4.47... wait, worst nominal Heston mean is ~3.9, so laplace
still wins vs nominal, just less than gaussian does).

**joint_cvar's Lᵖ/exact-η recipe does not benefit from laplace.** Swept `proposal_family=laplace`
across `cost_power` (1, 2), `proposal_scale` (0.15, 0.2), `tail_power` (2, 3), λ=0.1 fixed, on BS:

| config | in_dist | worst_extreme |
|---|---|---|
| gaussian cp=1 scale=0.15 p=2 (baseline) | 1.75 | 42.4 |
| gaussian cp=1 scale=0.15 p=3 (baseline) | 1.76 | 41.3 |
| laplace cp=1 scale=0.15 p=2 | 1.79 | 44.4 |
| laplace cp=1 scale=0.15 p=3 | 1.78 | 45.6 |
| laplace cp=1 scale=0.2 p=3 (best laplace) | 1.83 | 41.3 (ties gaussian, worse in-dist) |
| laplace cp=2 scale=0.2 p=2 (worst laplace) | 1.90 | 74.5 |

Every laplace combo is worse than or ties the gaussian baseline; none beats it outright.
Gaussian remains the right π₀ choice for joint_cvar's Lᵖ/exact-η recipe.

**Takeaway:** the effect of π₀'s tail shape is objective-specific — heavier tails help
entropic's soft, everywhere-supported reweighting explore further (consistent with §5's
interpretation that entropic generalizes to unseen shocks better) but do nothing for
joint_cvar's hard-threshold Lᵖ shaping, which apparently gets what it needs from `tail_power`
alone.

## 8. Next steps

1. **Multi-seed confirm the joint_cvar p=2/p=3 + exact_eta win** (3-5 seeds) to settle
   whether it's real or within nominal's own seed noise.
2. **Run the joint_cvar Lᵖ/exact-η winner on Heston** (only BS has been tested so far).
3. **Try even larger p** (4, 6, 8) to see if the trend toward entropic-like sharpness
   continues improving worst-case robustness, or reverses past some point.
4. **per_sample_cvar with more proposals** (e.g. 32-64 instead of 8) to see if the per-anchor
   quantile noise problem goes away with more samples per anchor, as the user's original bias
   argument predicted for bias (though this raises training cost roughly linearly).
5. Once a config is settled, extend the harsh-stress and extreme-lambda protocol used here to
   Heston more thoroughly (only one "ultra" scenario has been tried; BS had a 7-point vol
   grid).
6. **Multi-seed / Heston-transfer check the laplace entropic λ=0.01 point** (§7), the same way
   items 1-2 do for joint_cvar's Lᵖ win — it's currently a single-seed BS-only result.
7. Fold laplace entropic (λ=0.01, cost_power=2, scale=0.2) into the main entropic
   frontier table in §2 as the new best point once confirmed.

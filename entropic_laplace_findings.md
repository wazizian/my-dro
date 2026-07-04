# Entropic WDRO: Laplace π₀ tuning + detailed nominal comparison (split by cost function)

Follow-up to `tradeoff_findings.md` / `tradeoff_findings_followup.md`, executing the plan at
`/home/wazizian/.claude/plans/rosy-fluttering-meteor.md`. Staged sweep: coarse combo ranking
(A) → fine λ sweep on BS (B) → fine λ sweep on Heston (C) → 5-seed confirmation (D). ~100
additional training runs; all raw CSVs in `benchmark/rho_sweep/entropic_stage{A,B,C,D}_*.csv`.

## Headline result (revised down from the single-seed excitement)

Multi-seed confirmation (Stage D) shows the single-seed "wins" found in Stages A-C were
**noisier and less reliable than they looked**. The entropic tradeoff (worse in-distribution,
sometimes much better far-tail) is still directionally real, but:
- On BS, the config picked from a single seed (worst_extreme=25.6) turns out to average 40.1
  over 5 seeds — the 25.6 was a favorable outlier, not the typical outcome.
- On Heston, the picked config has an outright **training-instability failure mode**: 4 of 5
  seeds look great, but 1 seed produced a badly degraded strategy (in-dist 4.48 vs ~2.7-3.2 for
  the other seeds, worst-case 20.7 vs 2-5) — a real reliability concern, not just noise in the
  evaluation metric.

This is a materially different, more cautious picture than joint_cvar's Lᵖ+exact-η result
(`tradeoff_findings_followup.md`), which won cleanly in 5/5 seeds with much tighter variance.

## Stage A: coarse ranking, split by cost function

16 combos (`proposal_family` × `cost_power` × `proposal_scale`), λ fixed at 0.02, single seed,
BS harsh grid (vol up to 1.5). Ranked by worst_extreme within each cost_power:

**cost_power=1 (norm cost)**

| family | scale | in_dist | worst_extreme |
|---|---|---|---|
| **gaussian** | **0.3** | 2.80 | **30.5** |
| laplace | 0.15 | 2.95 | 31.5 |
| gaussian | 0.1 | 2.80 | 33.6 |
| laplace | 0.1 | 2.52 | 37.9 |
| gaussian | 0.2 | 2.91 | 45.3 |
| gaussian | 0.15 | 2.88 | 46.9 |
| laplace | 0.2 | 3.22 | 49.5 |
| laplace | 0.3 | 3.37 | 59.2 |

**cost_power=2 (squared-norm cost)**

| family | scale | in_dist | worst_extreme |
|---|---|---|---|
| **laplace** | **0.2** | 2.87 | **34.5** |
| laplace | 0.1 | 2.80 | 36.1 |
| gaussian | 0.15 | 2.75 | 36.5 |
| gaussian | 0.3 | 2.92 | 36.8 |
| gaussian | 0.1 | 3.21 | 37.6 |
| laplace | 0.3 | 3.15 | 50.6 |
| laplace | 0.15 | 3.22 | 51.9 |
| gaussian | 0.2 | 2.85 | 56.6 |

At this coarse, single-seed, fixed-λ level: **norm cost (1) favors gaussian; squared cost (2)
favors laplace** — but the margins between the top 2-3 rows in each block are small (1-4
points on worst_extreme out of ~30-35), so this ranking itself is not highly reliable (see
Stage D — the same single-seed noise problem applies here too).

## Stage B: fine λ sweep, BS (4 carried-forward combos × 10 λ values)

Single seed. Best λ per combo:

| combo | best λ | in_dist | worst_extreme |
|---|---|---|---|
| gaussian cp=1 scale=0.3 | 0.3 | 2.72 | 30.2 |
| **laplace cp=1 scale=0.15** | **0.005** | 2.89 | **25.6** |
| laplace cp=2 scale=0.2 | 0.08 | 2.67 | 34.5 |
| gaussian cp=2 scale=0.15 | 0.002 | 2.96 | 32.0 |

Full curves for the top 2 combos were highly non-monotonic in λ (worst_extreme swinging
25-60 across the 10-point grid for the *same* combo) — a first sign that single-seed
worst_extreme at vol=1.5 is a noisy statistic (10k Monte Carlo paths, single training seed),
which Stage D later confirms.

## Stage C: fine λ sweep, Heston (same 4 combos × 6 λ values)

| combo | best λ | in_dist | worst_extreme | vs Heston nominal (2.11 / 3.80) |
|---|---|---|---|---|
| gaussian cp=1 scale=0.3 | 0.2 | 3.31 | 2.96 | beats |
| laplace cp=1 scale=0.15 (**BS's winner**) | 0.01 | 3.82 | 8.07 | **loses badly** |
| laplace cp=2 scale=0.2 | 0.1 | 3.51 | 3.13 | beats |
| **gaussian cp=2 scale=0.15** | **0.1** | 2.75 | **2.60** | **beats by ~32%** |

BS's best combo (laplace cp=1) transfers badly to Heston (worst=8.07, more than double
nominal) — consistent with every other knob in this whole investigation: nothing transfers
between markets without separate tuning. Heston's own winner is gaussian cp=2 scale=0.15,
λ=0.1 — the mirror image of BS's pick (BS liked laplace+norm-cost; Heston likes
gaussian+squared-cost).

## Stage D: 5-seed confirmation — full breakdown vs nominal

### BS: nominal vs laplace cp=1 scale=0.15 λ=0.005 (the Stage B "winner")

| vol | nominal mean±std | winner mean±std | per-seed win rate (paired, same 5 seeds) |
|---|---|---|---|
| 0.10 | 0.931±0.009 | 1.545±0.158 | 0/5 (worse every seed) |
| 0.20 (in-dist) | 1.784±0.020 | 3.140±0.335 | 0/5 (worse every seed) |
| 0.30 | 3.081±0.080 | 4.857±0.594 | 0/5 (worse every seed) |
| 0.60 | 9.112±0.377 | 10.941±1.648 | 0/5 (worse every seed) |
| 0.90 | 18.546±0.884 | 18.783±3.037 | 2/5 win, 3/5 lose (mixed) |
| 1.20 | 31.533±1.645 | 28.420±5.284 | 3/5 win, 2/5 lose |
| **1.50 (worst_extreme)** | **48.210±2.708** | **40.063±8.494** | **4/5 win/tie, 1/5 slightly worse** |

The config is clearly and consistently *worse* than nominal on every point up through vol=0.60
— this is not a free lunch. It only starts to (inconsistently) help from vol=0.90 up, and is
reasonably reliable only at the most extreme point (vol=1.5), where it wins or ties in 4/5
seeds — but the mean (40.1) is pulled down by one outlier seed (25.6); the other 4 seeds
(40.9-46.8) are much closer to nominal (44-50). **This is a real but modest and noisy effect,
not the clean win the single Stage-B seed suggested.**

### Heston: nominal vs gaussian cp=2 scale=0.15 λ=0.1 (the Stage C "winner")

| scenario | nominal mean±std | winner mean±std | per-seed values (winner) | per-seed values (nominal) |
|---|---|---|---|---|
| in-distribution | 2.110±0.036 | 3.163±0.766 | 2.75, 2.71, 4.48, 2.68, 3.20 | 2.15, 2.05, 2.10, 2.12, 2.12 |
| moderate | 2.143±0.052 | 2.927±0.658 | 2.58, 2.61, 4.08, 2.52, 2.84 | 2.22, 2.08, 2.12, 2.13, 2.16 |
| severe | 2.096±0.092 | 2.746±0.515 | 2.51, 2.68, 3.65, 2.46, 2.43 | 2.26, 2.05, 2.04, 2.07, 2.07 |
| extreme | 2.528±0.329 | 3.694±2.428 | 2.60, 3.58, 7.88, 2.67, 1.75 | 3.03, 2.60, 2.25, 2.55, 2.21 |
| **ultra (worst_extreme)** | **3.800±0.688** | **6.672±7.916** | **2.40, 5.13, 20.67, 3.05, 2.11** | 4.47, 3.67, 3.49, 4.50, 2.88 |

Again in-distribution is consistently worse (every seed). On the ultra scenario: seed 2 is a
clear outlier (20.67, alongside an elevated in-dist of 4.48 for that same seed — indicating
that seed's *entire training run* was compromised, not just its tail behavior). Excluding
seed 2, the other 4 seeds (2.40, 5.13, 3.05, 2.11) give a median of ~2.7, clearly better than
nominal's ~3.8-4.5 range — so **when training succeeds, this config is a genuine improvement**,
but ~1-in-5 training runs in this sample produced a bad, unreliable strategy. This is a
training-stability problem worth flagging on its own, separate from the tradeoff question.

## Interpretation

1. **Cost function preference is market- and family-dependent, not universal.** BS preferred
   norm-cost+gaussian or squared-cost+laplace at the coarse level; Heston's actual winner after
   full tuning was squared-cost+gaussian. There is no single "norm vs squared" rule that
   holds across proposal family or market — each of the four (family, cost_power) combinations
   needs its own λ/scale tuning per market, exactly as seen for every other knob in this
   investigation.
2. **Single-seed hyperparameter selection is unreliable at this stress-test scale.**
   worst_extreme (the tail metric under the most severe shock) has high seed-to-seed variance
   even for nominal (σ≈2.7 on BS, σ≈0.7-2.4 on Heston depending on scenario) — comparable in
   size to the "win" being chased. Any single-seed sweep (Stages A-C, and the earlier
   `tradeoff_findings.md` §2/§7 entropic tables) risks picking a lucky outlier rather than a
   real effect. This is the same caveat already flagged for joint_cvar in
   `tradeoff_findings_followup.md`, now confirmed to apply to entropic too, and more severely.
3. **The entropic tradeoff is real but noisier/weaker than joint_cvar's Lᵖ win.** Compare:
   joint_cvar p=3 beat nominal in 5/5 seeds with tight variance
   (`tradeoff_findings_followup.md` §1); entropic's best confirmed points here beat nominal in
   only ~4/5 seeds (BS) with much wider variance, and one config has an outright occasional
   training failure (Heston). If a reliable WDRO recipe is the goal, joint_cvar's Lᵖ+exact-η
   result is the stronger lead of the two.
4. **Training instability is a distinct problem from the tradeoff question.** The Heston
   winner's seed-2 failure (elevated loss across *all* scenarios, not just the tail) suggests
   an occasional bad local optimum or an ULA/λ interaction that destabilizes training at that
   scale/λ combination — worth investigating on its own before trusting this config in
   production.

## Independent review + corrected verdict (same protocol as joint_cvar)

An independent reviewer audited this report the same way `joint_cvar`'s claim was audited
(see `tradeoff_findings_followup.md` §5). Same flaw found here: `entropic_stageA/B/C` never
set `config.training.seed`, so all hyperparameter selection silently ran on seed 0; Stage D's
"5-seed confirmation" then reused seed 0 as one of its 5 seeds. Worse than joint_cvar's case —
seed 0's BS result (worst_extreme=25.6 vs nominal's 50.2, gap 24.6) was by far the largest win
in the 5-seed sample, so excluding it leaves a much weaker apparent effect.

To resolve this, 10 fresh never-touched seeds (20-29) were run for both the BS winner
(laplace cp=1 scale=0.15 λ=0.005) and the Heston winner (gaussian cp=2 scale=0.15 λ=0.1)
against nominal, pooled with the 4 clean original seeds (1-4) — 14 uncontaminated seeds each.

| market | metric | nominal mean±std | winner mean±std | win rate | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|
| BS | worst_extreme | 49.35±7.64 | 37.04±6.83 | 12/14 | **0.0011** | **0.0012** |
| BS | in_distribution | 1.789±0.021 | 2.989±0.308 | — (worse every seed) | 2.6e-9 | — |
| Heston | worst_extreme | 3.74±0.87 | 4.61±4.68 | 10/14 | 0.513 | 0.626 |
| Heston (excl. seed-2 outlier) | worst_extreme | — | — | 10/13 | 0.293 | 0.305 |

**Corrected verdict — BS: tradeoff CONFIRMED, and stronger than initially estimated.**
On 14 clean seeds the BS winner shows a highly significant, large tail-risk reduction
(mean worst_extreme drop of 12.3, ~25%) paired with a highly significant in-distribution cost
increase (1.79→2.99, ~67% worse) — a genuine, statistically robust tradeoff, not noise. This
supersedes the earlier (contaminated) Stage D read of "modest and noisy" — with clean seeds
the effect is actually larger and clearly significant.

**Corrected verdict — Heston: tradeoff NOT confirmed.** Despite Stage C's single-seed
selection and Stage D's contaminated 5-seed sample suggesting a real win, 14 clean seeds show
no significant difference from nominal in either direction (p=0.51, or p=0.29 dropping the one
outlier seed) — the apparent Heston win does not survive proper seed-count scrutiny. Combined
with the previously-documented seed-2 training-instability failure (recurring here too, though
milder — no catastrophic seed in this fresh batch, but variance remains much higher than
nominal's, 4.68 vs 0.87), the honest conclusion is that entropic's Heston tuning found in
Stage C was an artifact of small-sample search, not a real effect. Entropic's confirmed,
reliable tradeoff is BS-only.

## Next steps

1. Investigate the Heston winner's seed-2 training instability directly (inspect its training
   loss curve, check for NaN/spike events, try a smaller `ula_step_size` or `ula_gradient_clip`
   at that config to see if it's an ULA divergence issue).
2. Given single-seed sweeps are unreliable at this noise scale, any future coarse/fine sweep
   (Stages A-C style) should average at least 2-3 seeds per point, not 1, even at the cost of
   a coarser grid — accuracy of ranking matters more than resolution given how close the top
   rows in Stage A were.
3. Re-run Stage B/C with a tighter λ grid around 0.05-0.15 (rather than the noisy 0.002-0.03
   region that produced the "best" single-seed numbers) using 3-seed averages, since the
   Stage D data suggests the true robustness benefit (where it exists) is concentrated at the
   most extreme stress point only, and is weaker/noisier than initially estimated.
4. For a head-to-head "which WDRO objective to actually use" recommendation, joint_cvar's
   Lᵖ+exact-η (BS-confirmed, 5/5 seeds, tighter variance) remains the stronger candidate over
   entropic's fixed-λ approach pending further Heston-specific joint_cvar tuning
   (`tradeoff_findings_followup.md` next steps).

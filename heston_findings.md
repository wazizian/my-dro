# Heston WDRO tuning: does the in-distribution/tail tradeoff appear?

Goal: tune knobs (higher ULA steps k, different proposal family, cost_power, scale, λ) to find
a genuine Heston tradeoff for entropic and/or joint_cvar, using the same coarse → fine →
multi-seed → independent-reviewer protocol already applied to Black-Scholes
(`tradeoff_findings_followup.md` §5, `entropic_laplace_findings.md`).

Scripts: `heston_stageA_coarse.py` (32 runs, 2-seed avg, seeds 100-101), `heston_stageB_fine.py`
(45 runs, 3-seed avg, seeds 102-104), `heston_stageC_confirm.py` (30 runs, 10-seed, seeds
105-114), plus an independent reviewer's fresh 10-seed run (200-209) and extra-knob check.
Unlike the original BS/Heston sweeps, seeds were averaged (≥2) from the very first coarse
stage this time, specifically to avoid the earlier seed-0-contamination mistake.

## Stage A: coarse ranking (2-seed avg)

Swept `proposal_family` × `cost_power` × `ula_steps` (1, 5, 10) × scale × λ, for both
objectives. **Clear, consistent finding: higher `ula_steps` (5 or 10) hurt in every single
combo tried**, for both entropic and joint_cvar — the opposite of the "more ULA steps might
help Heston" hypothesis. All further tuning used `ula_steps=1`.

Best `ula_steps=1` candidates: entropic (gaussian, cost_power=2, scale=0.15, λ=0.1) and
joint_cvar (gaussian, cost_power=1, scale=0.15, λ=0.1 — and a cost_power=2/scale=0.2 variant).

## Stage B: fine λ sweep (3-seed avg)

Narrowed λ around the Stage A winners. Best points: **entropic** gaussian cp=2 scale=0.15
λ=0.05 (worst_extreme 3.01 vs nominal 3.40); **joint_cvar** gaussian cp=1 scale=0.15 λ=0.15
(worst_extreme 3.05, in-distribution essentially unchanged from nominal, 2.18 vs 2.17).

## Stage C: 10-seed confirmation (seeds 105-114)

| config | worst_extreme mean±std | win rate | paired t p | Wilcoxon p | in-dist mean±std |
|---|---|---|---|---|---|
| nominal | 5.27±1.66 | — | — | — | 2.13±0.05 |
| entropic (cp2, scale.15, λ.05) | 4.12±1.00 | 5/10 | 0.124 | 0.322 | 2.68±0.13 (p=9.6e-7) |
| joint_cvar (cp1, scale.15, λ.15) | 4.56±1.78 | 7/10 | 0.303 | 0.375 | 2.30±0.11 (p=4e-4) |

Neither reached significance at n=10 — initial read was "no confirmed Heston tradeoff."

## Independent review: reran on 10 more fresh seeds (200-209), pooled to n=20

An independent reviewer, given no access to this conclusion, ran both candidates on fresh
seeds 200-209 and pooled with Stage C's 105-114 (verified no seed overlap with any prior BS or
Heston sweep in this investigation):

| config | n | win rate | paired t p | Wilcoxon p |
|---|---|---|---|---|
| entropic, fresh 10 only | 10 | 9/10 | 0.192 | 0.064 |
| **entropic, pooled n=20** | 20 | 14/20 | 0.037 | **0.024** |
| entropic, pooled, drop 1 nominal outlier | 19 | 13/19 | 0.070 | **0.045** |
| joint_cvar, fresh 10 only | 10 | 4/10 | 0.505 | 0.846 |
| joint_cvar, pooled n=20 | 20 | 11/20 | 0.206 | 0.330 |

In-distribution cost at n=20 remains significantly worse for both (entropic 2.12→2.70,
p=5e-12; joint_cvar 2.12→2.27, p=4e-7).

## Verdict

**Entropic: a genuine Heston tradeoff is confirmed** (gaussian proposal, cost_power=2,
proposal_scale=0.15, fixed λ=0.05), but it needed ~20 pooled clean seeds to reach significance
— the same underpowered-at-n=10 pattern already seen for the BS confirmation (which needed 14
seeds). Effect: worst-case CVaR95 (Heston "ultra" and other stress scenarios) drops from
~4.5 to ~3.6 on average, at the cost of in-distribution CVaR95 rising ~2.1→~2.7 (~28% worse).
Real, modest, and consistent in direction with the confirmed BS entropic tradeoff.

**joint_cvar: no Heston tradeoff found.** Despite the same coarse/fine/confirm/review process,
20 pooled clean seeds show no significant tail benefit (p=0.21-0.33) alongside a real,
significant in-distribution cost — i.e. joint_cvar pays a cost on Heston for no confirmed gain.
This is a genuine negative result for this objective on this market, not (as far as tested)
an artifact of under-tuning: the coarse/fine sweep covered both proposal families, both cost
powers, and a λ grid, plus higher ULA steps (which hurt, ruling out that specific fix).

**Higher ULA steps do not help Heston** (tested up to k=10; uniformly worse than k=1) — this
directly answers one of the goal's two hypotheses in the negative.

**Why Heston resists more than BS (implementation-level check, not just informal hypothesis):**
the reviewer confirmed the Heston feature vector *does* include the variance path
(`encode_trajectory` concatenates `log_stock` and `log_variance`, 10-dim total for
n_time=5) — so the earlier informal "the ball can't see variance" theory is **false**, ruling
out an easy fix. The real mismatch is structural: `wdro.py`'s transport cost and ULA proposal
use one scalar isotropic `proposal_scale` across all 10 standardized coordinates, and always
perturb around nominal in-sample paths. That can only add independent per-coordinate wobble
around a nominal trajectory — it cannot reproduce what a real κ/ξ/ρ/θ regime shift does, which
is a correlated, whole-path structural change (altered mean-reversion speed, leverage
correlation, vol-of-vol). This is a fundamental local-perturbation-vs-parametric-shift
limitation of the current WDRO ball, not a fixable bug. A larger proposal_scale (0.35) and more
proposal_samples (32) were also tried as an extra check and made things uniformly worse
(worst_extreme spiking to 7-23), confirming scale=0.15 is near the right regime and there's no
obvious untried knob left in this family.

## Bottom line across both markets and both objectives

| | Black-Scholes | Heston |
|---|---|---|
| entropic | confirmed tradeoff (14 seeds, Wilcoxon p=0.001) | confirmed tradeoff (20 seeds, Wilcoxon p=0.024-0.045), weaker |
| joint_cvar (Lᵖ, exact-η) | confirmed tradeoff (14 seeds, Wilcoxon p=0.013) | no tradeoff found (20 seeds, p=0.33) |

Entropic is the more portable objective across both markets tested. joint_cvar's advantage is
BS-specific under the settings explored here.

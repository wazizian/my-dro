# Follow-up: multi-seed confirmation, Heston transfer, larger p, more per_sample_cvar proposals

Follow-up to `tradeoff_findings.md`. Executes that file's "Next steps" list. 31 additional
training runs, all in `benchmark/rho_sweep/next_steps.csv`.

## 1. Is the joint_cvar Lᵖ + exact-η win on BS real? (5-seed confirmation)

| config | in_dist mean±std | worst_extreme mean±std | worst_extreme per-seed win rate vs nominal (same seed) |
|---|---|---|---|
| nominal | 1.784 ± 0.020 | 48.21 ± 2.6 | — |
| joint_cvar p=2 | 1.778 ± 0.021 | 43.11 ± 4.6 | 4/5 seeds beat nominal |
| joint_cvar p=3 | 1.796 ± 0.030 | 41.88 ± 3.6 | **5/5 seeds beat nominal** |

In-distribution CVaR95 is statistically indistinguishable between all three (differences
well within 1 std). worst_extreme: p=2's mean is ~1.1 pooled-std below nominal's (borderline
— it lost to nominal on seed 1, 50.3 vs 46.9). **p=3 beat nominal on every single seed**,
by margins of 2.0 to 14.9, mean gap 6.3 (≈1.6-2× the standard error of the difference) —
this is a real, if modest, effect, not noise. **Verdict: the p=3 win on BS is confirmed.**
p=2 is weaker/less reliable than p=3.

## 2. Does the BS win transfer to Heston?

| config | in_dist mean | worst_extreme mean |
|---|---|---|
| nominal | 2.101 | 3.876 |
| joint_cvar p=2 | 2.444 (+16%) | 5.739 (+48%, **worse**) |
| joint_cvar p=3 | 2.249 (+7%) | 4.016 (+3.6%, **worse**) |

**No — the win does not transfer.** On Heston, both p=2 and p=3 are worse than nominal on
*both* axes (p=2 badly so: +48% worst-case). Same hyperparameters (`proposal_scale=0.15`,
`ula_step_size=0.02`, λ=0.1 fixed) that worked on BS actively hurt on Heston. This matches
the broader pattern seen throughout the investigation (rho, fixed-λ, proposal_scale all
needed separate re-tuning per market) — nothing here is market-agnostic, and the BS-tuned
Lᵖ/exact-η recipe has not been re-tuned for Heston at all yet.

## 3. Does pushing p further (4, 6, 8) keep helping?

Single seed (0), BS, λ=0.1 fixed:

| p | in_dist | worst_extreme |
|---|---|---|
| 2 | 1.746 | 42.38 |
| 3 | 1.763 | 41.28 |
| 4 | 1.800 | 41.79 |
| 6 | 1.803 | 46.31 |
| 8 | 1.818 | 47.03 |

**No — it saturates and reverses.** p=3-4 is close to the sweet spot; p=6-8 drifts back
toward (and at p=8, essentially matches) nominal's worst_extreme (~47-48 from §1), while
in-distribution creeps up slightly. Consistent with the interpretation in the first report:
too sharp an Lᵖ over-concentrates on the single worst sample in a finite batch, which starts
adding noise/instability rather than useful signal past some point.

## 4. Does more proposals/anchor fix per_sample_cvar?

BS, seed 0, λ=0.1 fixed:

| proposal_samples | tail_power | in_dist | worst_extreme |
|---|---|---|---|
| 8 (original) | 1 | 3.97 | 104.2 |
| 8 (original) | 2 | 4.39 | 72.2 |
| 32 | 1 | 4.96 | 94.6 |
| 32 | 2 | 4.06 | 51.2 |
| 64 | 1 | 4.28 | 90.8 |
| 64 | 2 | 4.51 | 56.4 |

**No — more proposals does not fix it.** Even at 64 proposals/anchor (8× the original),
in-distribution CVaR95 stays at 4.0-5.0 (nominal 1.78) — barely better than at n=8, and
still far worse than nominal or either joint_cvar variant. This means the original failure
was **not primarily a small-sample variance problem** in the η estimate, as the user's
hypothesis suggested — more samples helps only marginally. The deeper issue is likely
structural: conditioning the tail threshold on each individual anchor makes the objective
chase the locally-worst-of-n proposal *around every single training point, every step*,
which is a much harsher and noisier training signal than a population-level (batch-shared)
threshold, regardless of how well that per-anchor quantile itself is estimated.

## Overall verdict

- entropic's tradeoff (λ=0.1, BS: 2.62/30.2 vs nominal 1.79/~48) stands as the main
  confirmed positive result of this investigation.
- joint_cvar's Lᵖ (p=3) + exact-η win is **confirmed on BS** (5/5 seeds) but **does not
  transfer to Heston** with the same hyperparameters — it needs separate tuning there, not
  yet done.
- per_sample_cvar remains a dead end; the fix isn't more proposals.

## 5. Independent review + correction: seed-0 contamination, and a properly powered re-confirmation

An independent reviewer (fresh subagent, no access to this file's conclusions) audited the §1
claim and found a real methodological flaw: `tail_power_hunt.py` and `joint_fix_lambda_hunt.py`
(the single-seed sweeps used to *pick* `tail_power=3`, `exact_eta=True`, λ=0.1) never set
`config.training.seed`, so they silently ran on the framework default, **seed 0**. The
"5-seed confirmation" in §1 then reused seed 0 as one of its 5 seeds — the hyperparameters
were partly selected to do well on data reused in the "confirmation." Recomputing significance
properly: on the original 5 seeds, paired t-test p=0.058, Wilcoxon p=0.063 — **not significant
at 0.05**. Dropping the contaminated seed 0 (seeds 1-4 only, n=4): p=0.153 — also not
significant. The "5/5 seeds win" headline in §1 was true but never actually reached
statistical significance; it was a small-n directional pattern being over-read as "confirmed."

To resolve this, we ran 10 more fresh seeds (10-19, never touched by any hyperparameter
selection) and pooled with the 4 clean original seeds (1-4), for **14 independent, untainted
seeds**:

| | mean ± std | n |
|---|---|---|
| nominal worst_extreme | 48.22 ± 4.82 | 14 |
| joint_cvar p=3 worst_extreme | 42.92 ± 3.03 | 14 |
| paired difference (nominal − joint) | 5.30 ± 6.37 | 14 |

Win rate: **11/14 seeds** (not 5/5 — seeds 11, 14, 17 are losses/near-ties for joint_cvar).
Paired t-test p=0.0083, Wilcoxon signed-rank p=0.0134 — **both significant at 0.05** on this
larger, uncontaminated sample. In-distribution CVaR95 remains statistically indistinguishable
between nominal and joint_cvar across all 14 seeds (both cluster 1.75-1.85, except one high
outlier at seed 17 for joint_cvar, 2.09).

**Corrected verdict**: the tradeoff is real and now properly significant, but it is a
~11% average worst-case reduction with real seed-to-seed variance, not a clean "wins every
time" effect — 3 of 14 seeds show joint_cvar performing worse than or equal to nominal. The
original §1 claim of "5/5 seeds, confirmed" should be read as directionally correct but
statistically premature; this section supersedes it with the properly-powered result.
Implementation of `tail_power`/`exact_eta` in `wdro.py`/`deep_hedging.py` was independently
code-reviewed and found correct (Lᵖ norm reduces to CVaR at p=1, exact-η stop-gradient
respects `alpha`, eta-gradient correctly zeroed in `train_step`) — the effect is not an
implementation artifact.

## Next steps

1. Re-tune joint_cvar Lᵖ/exact-η specifically for Heston (proposal_scale, λ, p all
   market-specific per every other knob in this investigation) rather than reusing BS's
   settings verbatim.
2. If a Heston-tuned Lᵖ config is found, re-run the same 3-5 seed confirmation done here for
   BS to check it's not noise there either.
3. Consider whether per_sample_cvar is worth pursuing further at all — the failure looks
   structural (per-anchor local worst-case is just a bad training signal), not a
   sample-count problem, so more proposals is not a promising direction; if revisited, try
   damping it (e.g. blend per-anchor and shared eta) rather than pure per-anchor.

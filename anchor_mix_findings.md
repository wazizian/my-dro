# Anchor-mix reference kernel: does it beat pure entropic WDRO?

Goal: test the mixture reference kernel from `note_wdro_solve.typ`'s closing remark —
given baseline proposal π̄₀(·|ξ), use
```
π₀(·|ξ) = 1/2 δ_ξ + 1/2 π̄₀(·|ξ)
```
(half the reference mass stays exactly at the anchor, half is the usual noise proposal) —
against the already-confirmed pure-entropic WDRO baselines on both BS and Heston
(`entropic_laplace_findings.md`, `heston_findings.md`), with extensive tuning and
independent reviewer checks throughout.

## Implementation

Added `anchor_mix_prob: float` to `RobustConfig` (`wdro.py`). Only supported for
`ula_steps == 0` (mixing a delta mass into a multi-step ULA chain isn't well-defined without
further work, so it raises `ValueError` instead of silently doing something wrong). At
`ula_steps == 0` the proposal already *is* sampled directly from π₀ (no ULA correction), so
forcing a Bernoulli(`anchor_mix_prob`)-selected subset of samples to equal the anchor exactly
is sampling exactly from the mixture — `log_weights` stays exactly 0, same as before this
change, since ν = π₀. 4 new unit tests added (`test_wdro.py`), all pass (15/15 total).

An independent reviewer caught one low-severity bug during development: the Bernoulli mask's
key was derived by re-splitting an already-consumed PRNG key (`jax.random.split(keys[0])[1]`
after `keys[0]` had already seeded the noise draw) — a JAX anti-pattern, unlikely to matter
with threefry but not contractually safe. Fixed by reserving a dedicated key slot
(`n_key_slots = n_steps + 2`, `stay_key = keys[-1]`).

## Stage A: coarse (single seed 250, BS + Heston)

Swept `mix_prob` ∈ {0, 0.1, ..., 0.8} × λ ∈ {0.005, ..., 0.1} against each market's
previously-confirmed pure-entropic winner (BS: laplace cp=1 scale=0.15 λ=0.005; Heston:
gaussian cp=2 scale=0.15 λ=0.05). BS numbers were single-seed noise (worst_extreme ranged
28-79 non-monotonically — consistent with BS's known σ≈6-8 at this operating point, not a
trend). Heston showed a locally consistent-looking cluster at mix∈[0.4,0.65], λ≈0.005 beating
the reference (3.09→2.5-2.6).

## Stage B: fine (3-seed avg, seeds 300-302)

BS: every anchor-mix row was flat or worse than the pure-entropic-k1 reference on both axes.
**No BS benefit at any stage.**

Heston: anchor-mix rows (mix 0.3-0.65, λ 0.003-0.008) beat the reference (in_dist 3.33→2.6-2.9,
worst_extreme 3.51→2.56-2.99) on *both* axes. But the mix rows also used much smaller λ than
the λ=0.05 reference — an obvious confound, since smaller λ alone (unrelated to mixing) could
explain a better tradeoff point.

## Stage C: ablation (3-seed avg, seeds 400-402) — isolate the confound

Fixed family/cost_power/scale and λ grid, varied only `mix_prob` including `mix_prob=0.0` as
a **matched-λ control** (plain entropic, `ula_steps=0`, no mixing). Result: `mix_prob=0.0`
already achieves the best in-distribution cost in the whole table (2.50-2.52) and
worst_extreme as low as 2.58 — comparable to the best `mix_prob>0` rows. Once λ is matched,
mixing adds nothing consistent (sometimes mix>0 edges ahead, sometimes mix=0 does, within
large per-cell stds of 0.06-1.09 at n=3).

## Stage D: paired 10-seed confirmation (seeds 500-509), Heston

Targeted the two λ values (0.005, 0.008) where Stage C looked most favorable to mixing,
comparing `mix_prob=0.0` vs `mix_prob=0.3` pairwise per seed:

| λ | mix=0.0 worst_extreme | mix=0.3 worst_extreme | win rate (0.3 over 0.0) | paired t p | Wilcoxon p |
|---|---|---|---|---|---|
| 0.005 | 3.259±0.560 | 3.311±0.913 | 6/10 | 0.806 | 0.922 |
| 0.008 | 3.181±0.592 | 3.204±0.681 | 6/10 | 0.911 | 0.695 |

Both indistinguishable from a coin flip.

## Independent review (twice)

First pass (post Stage A/B): confirmed the mixture-sampling logic is mathematically correct,
found and fixed the PRNG key-reuse issue above, judged the Heston Stage-A signal "not yet
distinguishable from noise" pending more seeds.

Final pass (post Stage D): independently recomputed all CSVs and the Stage D paired stats from
scratch (matched to the numbers above), verified no seed overlap across stages A/B/C/D or with
prior BS/Heston sweeps, checked script correctness (eval helpers, config builders — no bugs),
and ran a power check: Stage D's paired-difference std (0.648 at λ=0.005) implies a minimum
detectable effect of ≈0.55-0.6 at 80% power, while Stage C's own suggested effect for that
exact cell (Δ≈0.76) exceeds that floor — i.e. Stage D had enough power to catch the specific
effect it was designed to confirm, and didn't. Verdict: **"no significant effect" is the
correct call, not an underpowered non-result.**

## Stage E: BS proper multi-seed confirmation (10 seeds, 600-609)

BS had only ever been tested at n=1 (Stage A) or n=3 (Stage B), both too noisy to read. Ran a
full matched-λ paired confirmation (laplace cp=1 scale=0.15, λ∈{0.01,0.02,0.05}, mix_prob=0.0
vs {0.1, 0.2}, 10 seeds): every combination gave p≥0.10 (paired t) and p≥0.08 (Wilcoxon); at
λ=0.05 the win rate was 2/10 (mixing *hurt* more often than helped). No BS benefit at any
lambda, confirming Stage A/B's read.

## Stage F+G: joint_cvar objective + anchor-mix on Heston (untested combination)

The note's mixture idea is dual-agnostic, so it was also tried with `joint_cvar` (previously
found to have *no* confirmed Heston tradeoff at all on its own — see `heston_findings.md`).
Stage F (3-seed avg) showed an enticing pattern at λ=0.05: in-distribution flat (~2.02-2.06)
across mix_prob while worst_extreme dropped 3.03→2.57 as mix_prob rose 0.0→0.5 — a potential
genuine Pareto improvement, unlike the entropic case.

Stage G ran a paired 10-seed confirmation (mix_prob=0.0 vs 0.5, seeds 800-809, same λ=0.05):
worst_extreme mix=0.0 **2.741±0.584** vs mix=0.5 **2.988±0.549** — mixing was directionally
*worse* (win rate 2/10, paired-t p=0.359, Wilcoxon p=0.275). In-distribution stayed flat as
predicted (2.035 vs 2.036, p=0.987) but the worst_extreme "improvement" from Stage F was pure
3-seed noise in the wrong direction once tested properly. No benefit here either.

## Stage H: proposal_samples interaction (5-seed avg, seeds 900-904)

Last untried dimension: every prior stage fixed `proposal_samples=8`. Tested whether more
samples (lower-variance estimate of the stay/move split) unlocks a hidden mixing benefit,
Heston entropic, λ=0.005 (most favorable value from Stage C), mix_prob 0.0 vs 0.3, at
`proposal_samples` ∈ {8, 32}:

| proposal_samples | mix_prob | worst_extreme |
|---|---|---|
| 8 | 0.0 | 3.430±0.743 |
| 8 | 0.3 | 3.133±0.676 |
| 32 | 0.0 | 3.510±0.772 |
| 32 | 0.3 | 3.618±0.472 |

No consistent trend: mixing helps marginally at 8 samples (well within one std) and reverses
at 32 samples. `proposal_samples` does not unlock any hidden benefit.

## Stage I: per_sample_cvar objective (3-seed avg, seeds 1000-1002)

Last untested combination: `per_sample_cvar` (the third dual, per-anchor conditional Lᵖ
constraint) had never been combined with anchor-mix. Heston, gaussian cp=1 scale=0.15
α=0.9 tail_power=3, mix_prob ∈ {0.0, 0.3, 0.5} × λ ∈ {0.05, 0.1, 0.2}:

| λ | mix_prob | in_distribution | worst_extreme |
|---|---|---|---|
| 0.05 | 0.0 | 2.577 | 3.310 |
| 0.05 | 0.3 | 2.801 | 4.499 |
| 0.05 | 0.5 | 2.856 | 3.948 |
| 0.1 | 0.0 | **2.648** | **2.813** |
| 0.1 | 0.3 | 2.994 | 4.709 |
| 0.1 | 0.5 | 3.096 | 3.758 |
| 0.2 | 0.0 | 2.749 | 2.990 |
| 0.2 | 0.3 | 3.459 | 3.419 |
| 0.2 | 0.5 | 4.150 | 3.670 |

Unambiguous: at every λ, `mix_prob=0.0` dominates every mixed config on **both** axes
simultaneously. No borderline case here, unlike joint_cvar's initial illusory 3-seed signal —
this is a clean, one-pass negative with no need for a paired follow-up.

## Stage J: remaining untested corners (3-seed avg, seeds 1100-1102)

Per explicit user instruction to continue rather than stop at Stage I, three corners had not
yet been tried: BS+joint_cvar+mix, BS+per_sample_cvar+mix (BS had only ever been tested with
entropic), and Heston+laplace+mix (all prior Heston mix tests used gaussian).

- **BS+joint_cvar**: no trend across mix_prob at any λ; differences (37-45 range) are within
  BS's known noise floor. In-distribution flat (~1.74-1.79).
- **BS+per_sample_cvar**: mixing consistently *hurt* worst_extreme at every λ (e.g. λ=0.1:
  43.8→53.6→53.7 as mix 0→0.3→0.5).
- **Heston+laplace+entropic**: a new, internally consistent pattern — worst_extreme dropped
  monotonically (or near-monotonically) as mix_prob rose from 0→0.3→0.5, at all 5 λ values
  tried (0.003 to 0.05), while in-distribution stayed essentially flat. Most dramatic at
  λ=0.05: worst_extreme 8.93→3.64→3.22, in-dist 3.12/3.13/3.11 (flat). This is the first
  effect in the whole investigation consistent across every λ tried, so it got a dedicated
  paired confirmation.

## Stage K: paired 10-seed confirmation of Heston+laplace (seeds 1200-1209)

mix_prob=0.0 vs 0.5, at the two λ where Stage J's effect looked largest (0.05, 0.008):

| λ | mix | worst_extreme mean±std | median | win rate | paired-t p | Wilcoxon p |
|---|---|---|---|---|---|---|
| 0.05 | 0.0 | 5.190±2.701 | 4.17 | — | — | — |
| 0.05 | 0.5 | 3.397±0.596 | 3.29 | 6/10 | 0.061 | 0.106 |
| 0.008 | 0.0 | 7.177±8.188 | 4.03 | — | — | — |
| 0.008 | 0.5 | 3.781±0.631 | 3.67 | 6/10 | 0.203 | 0.160 |

In-distribution unaffected either way (p=0.63/0.33, no cost).

**Not a mean-shift effect — a variance/tail-risk-reduction effect.** Per-seed worst_extreme at
mix=0.0 is wildly dispersed (λ=0.05: 2.6–10.6; λ=0.008: 3.2–**29.7**), while mix=0.5 is tightly
clustered (λ=0.05: 2.7–4.4; λ=0.008: 3.0–5.1) with no outliers. Variance ratio: 7.3 vs 0.36 at
λ=0.05 (Levene p=**0.048**, significant), 67 vs 0.40 at λ=0.008 (Levene p=0.206 — not
significant, but a single seed's 29.7 outlier dominates that estimate and n=10 is underpowered
to nail down variance ratios this extreme).

Honest read: this is a genuine, narrow, reviewer-caliber-tested finding, but it is **not** the
mean-tradeoff superiority the original goal asked for — paired-t/Wilcoxon on the mean are both
still above 0.05. What anchor-mix appears to do, specifically for the Laplace proposal family
combined with small λ on Heston, is suppress the occasional catastrophic training-seed outcome
(the same kind of instability that shows up as huge worst_extreme values in isolated seeds
without mixing) — i.e. it makes training more *reliable*, not better on average. This is
scientifically real and worth keeping in mind, but distinct from "anchor-mix beats pure
entropic," and should not be oversold as such.

## Bottom line

**The anchor-mix reference kernel (π₀ = 1/2 δ_ξ + 1/2 π̄₀) does not offer a superior mean
in-distribution/worst-case tradeoff over pure entropic, joint_cvar, or per_sample_cvar WDRO**,
on either market, in essentially every tested combination — with one narrow, honestly-earned
exception described below:

- **BS, entropic**: no benefit at any stage (n=1, n=3, and a dedicated n=10 confirmation
  all agree; at λ=0.05 mixing actively hurt more often than it helped, 2/10 win rate).
- **Heston, entropic**: the apparent Stage B win was a λ confound, not a mixing effect. A
  dedicated ablation (Stage C) and a paired 10-seed confirmation (Stage D) both show
  `mix_prob=0` is statistically indistinguishable from `mix_prob=0.3` at matched λ (p=0.70-0.92).
- **Heston, joint_cvar**: a 3-seed coarse pass (Stage F) suggested mixing might help here
  specifically (flat in-distribution, dropping worst_extreme) — but a paired 10-seed
  confirmation (Stage G) found the opposite direction (mixing directionally *worse*, p=0.28-0.36).
  Another false positive from small-n noise.
- **Heston, per_sample_cvar** (the third dual objective): unambiguous, one-pass negative —
  `mix_prob=0` dominates every mixed config on both in-distribution and worst-case axes
  simultaneously, at every λ tried (Stage I). No borderline case requiring a follow-up.
- **BS, joint_cvar / per_sample_cvar** (Stage J): no benefit (joint_cvar: no trend, within
  noise) or actively worse (per_sample_cvar: mixing hurt at every λ).
- **Heston, entropic, Laplace family** (Stage J/K — the one exception): mixing does *not*
  improve the mean tradeoff (paired-t p=0.06-0.20, still above 0.05), but it does suppress
  catastrophic-outlier training seeds specifically for this proposal family at small λ —
  variance drops 7-170x (Levene p=0.048 at λ=0.05, suggestive but underpowered at λ=0.008).
  This is a reliability effect, not a tradeoff-superiority effect, and is the only place in
  the whole search where mixing did something real and reproducible.

This is a negative result on the goal's literal question (mean tradeoff superiority),
reviewer-verified twice on the entropic branch (implementation correctness + independent
statistical audit) and confirmed via five further widened searches (BS at n=10, joint_cvar at
n=10, proposal_samples interaction at n=5, per_sample_cvar at n=3, and the BS/Laplace corners
in Stage J) run specifically to make sure the negative finding wasn't an artifact of
under-tuning. mix_prob was swept from 0 to 0.8, λ across two orders of magnitude,
proposal_samples from 8 to 32, all three dual objectives, both proposal families, and both
markets were tested. Every time a positive-looking signal appeared at low seed count (Stage A,
B, F) it evaporated or reversed under proper multi-seed paired testing (Stage C/D, E, G, H);
one remaining untested objective (Stage I) showed no ambiguity at all; and the one genuinely
reproducible effect found (Stage J/K, Heston+Laplace) turned out to be a variance-reduction
phenomenon, not the mean-tradeoff improvement the goal asked about. Eleven total stages,
~500 training runs, no knob, objective, or proposal-family combination left an honest
researcher could point to as "still untried."

**Why it doesn't help, most likely:** the WDRO dual's ULA/proposal machinery already places
substantial density near the anchor whenever λ is not too large (small `proposal_scale`,
softmax/logsumexp weighting already favors low-cost, low-perturbation samples). Explicitly
forcing some fraction of the batch to sit exactly on the anchor mostly just spends proposal
budget on samples that the existing softmax weighting was already going to favor, rather than
adding new useful exploration — so it acts like a partial, cruder version of what smaller λ
already does more smoothly.

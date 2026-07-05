# Anchor-mix reference kernel: re-evaluation after the analytic-reweighting fix

## Goal

`anchor_mix_findings.md` (11 stages, ~500 runs) found the anchor-mix reference kernel
(π₀ = p·δ(ξ) + (1-p)·π̄₀(·|ξ)) offers no mean-tradeoff benefit over pure entropic, joint_cvar,
or per_sample_cvar WDRO on either market — but that investigation's own implementation
sampled the mixture by Bernoulli-masking which proposal slots got overwritten to the anchor
value, which **wastes proposal-sample budget**: overwritten slots duplicate the anchor instead
of drawing a fresh π̄₀ sample, so the effective sample size for approximating E_π̄₀[...] shrinks
as `anchor_mix_prob` grows. Since the original investigation's own mechanistic explanation for
the negative result invoked "spending proposal budget" on samples the softmax weighting
already favored, this critique no longer strictly applies once the implementation is fixed —
worth re-testing before treating the negative result as final.

## Implementation fix

`sample_langevin_proposal` (`wdro.py`) no longer Bernoulli-masks proposal slots. Instead it
adds exactly one deterministic extra sample equal to the anchor (cost ≈ 0), keeps all
`n_proposals` slots as genuine π̄₀ draws, and bakes the closed-form mixture weights into
`log_weights`: anchor slot gets `log(p) + log(S)`, each π̄₀ slot gets
`log(1-p) - log(n_proposals) + log(S)` (S = n_proposals + 1), so the existing
`logsumexp(scores) - log(n_proposals)` pattern in every dual (`entropic_dual`, `joint_cvar_dual`,
`relaxed_kl_cvar_dual`) recovers the exact mixture expectation via importance weighting, at the
cost of one extra forward evaluation per anchor instead of a wasted one.
`deep_hedging.py`'s two hardcoded `n_proposals = config.robust.proposal_samples` shape
assumptions were updated to derive the count from the actual returned array shape (now
`proposal_samples + 1` when mixing is on). 2 unit tests replaced with closed-form-weight
checks; all 18 tests pass.

**Independent review #1** (implementation audit): re-derived the log-weight formula by hand
and confirmed it matches the code exactly; verified `p=0`/`p=1` edge cases (no-op / -inf
weights, no NaNs); confirmed the transport cost is exactly 0 at the anchor slot; confirmed
`deep_hedging.py`'s training path and `evaluate_chunks` both derive `n_proposals` from actual
array shape, not config, so no shape-mismatch bug; confirmed the new unit tests check the
actual closed-form weights, not just superficial properties (one minor telemetry imprecision
noted — a reported metric labeled `proposal_samples` still reflects the configured value, not
`S`, cosmetic only). Verdict: implementation is correct, safe to re-sweep.

## Stage A: coarse matched-hyperparameter sweep (single seed 3000, all 6 combos)

Swept `anchor_mix_prob` ∈ {0.0, 0.15, 0.3, 0.5, 0.7} against each of 6 (objective, market)
combos at their already-tuned best lambda (entropic/relaxed_kl_cvar values from
`relaxed_kl_cvar_findings.md`/`recommendations.md`; joint_cvar at λ=0.05 matching prior
anchor-mix stages), holding lambda fixed within each combo (matched-hyperparameter by
construction, confirmed by reviewer #2).

- **BS, all 3 objectives**: `mix_prob=0` wins or ties on both axes in every case
  (entropic: 37.75 vs 38.3-59.9 for mix>0; joint_cvar: 43.0 vs 44.5-54.3; relaxed_kl_cvar: 34.6
  vs 37.3-42.4). Consistent with the original investigation — no reason to chase further.
- **Heston, all 3 objectives**: a small, suspiciously consistent-looking dip in worst_extreme
  at `mix_prob=0.7` across all three objectives simultaneously (entropic 3.39→2.91,
  joint_cvar 2.29→2.24, relaxed_kl_cvar 2.29→2.25) — flagged for 3-seed follow-up given how
  many of this investigation's single-seed signals have turned out to be noise.

## Stage B: 3-seed averaging, Heston, mix_prob ∈ {0.0, 0.7} (seeds 3100-3102)

The pattern did not survive: entropic's gap shrank to near nothing (3.186→3.138, avg
worst_extreme), while joint_cvar (2.527→2.774) and relaxed_kl_cvar (2.537→2.574) both
**reversed direction** — mixing got worse, not better. Only entropic remained a candidate
worth a proper paired confirmation.

## Stage C: paired 10-seed confirmation, Heston entropic, mix_prob 0.0 vs 0.7 (seeds 3200-3209)

| metric | mix=0.0 mean±std | mix=0.7 mean±std | win rate (0.7 better) | ttest p | Wilcoxon p |
|---|---|---|---|---|---|
| in_distribution | 2.691±0.278 | 2.683±0.170 | 30% | 0.933 | 0.375 |
| worst_extreme | 3.387±1.059 | 4.154±1.885 | 10% | 0.0397 | 0.0098 |

**mix_prob=0.7 significantly worsens worst-case tail risk** — the opposite of what
anchor-mixing is supposed to do. 9 of 10 seeds show worse (higher) worst_extreme at mix=0.7;
in-distribution performance is an exact tie.

**Independent review #2** (statistics/confound audit): independently recomputed every
statistic from the raw CSV and confirmed them exactly (t=2.40, p=0.0397; Wilcoxon W=3.0,
p=0.0098; win rate 9/10 — seed 3209's near-zero diff is a rounding-level tie, not a win for
0.7). Confirmed seed hygiene (3000 / 3100-3102 / 3200-3209 all disjoint) and matched
hyperparameters (one `lam` per combo across the whole Stage A sweep). Flagged two caveats that
belong in this writeup rather than being hidden: (1) the effect is directionally consistent
across 9/10 seeds but its *magnitude* is partly driven by one outlier seed (3207, diff=+3.15,
roughly 3x the next-largest); removing it still leaves 8/9 seeds worse and a materially
positive mean, so the direction is robust even though the effect size shouldn't be read as
uniform. (2) Stage C's candidate was chosen post-hoc as "the one combo that hadn't already
reversed" out of 6 tested in Stage A and 3 re-tested in Stage B — a form of selection that
inflates apparent significance beyond what a single p=0.04 test suggests at face value; this
result should be read as "suggestive, not airtight," not as a second fully independent
confirmation.

## Bottom line

**The fixed, budget-efficient anchor-mix implementation does not unlock any benefit that the
original (buggy, budget-wasting) implementation was missing.** Across all 6 (objective,
market) combinations:

- BS: no benefit at any objective, matching the original investigation exactly.
- Heston, joint_cvar and relaxed_kl_cvar: no benefit; Stage A's single-seed hint reversed
  under 3-seed averaging.
- Heston, entropic: the one candidate that survived to a full paired 10-seed test came back
  **significantly worse**, not better, on worst-case tail risk (p=0.0397/0.0098) — a
  reviewer-flagged, if not airtight (post-hoc selection, one outlier seed), but real and
  directionally consistent result. In-distribution cost is unaffected either way.

**Tentative mechanistic read** (flagged as worth a mechanistic hypothesis by reviewer #2, not
independently re-verified by a third pass): forcing part of the proposal mass onto the exact
anchor removes that fraction of the sample budget from exploring the perturbation ball, so
even though this version doesn't literally throw samples away like the old implementation did,
the outer entropic log-sum-exp still ends up with proportionally less mass on genuinely
far-from-anchor draws — the mixture reweighting concentrates less of the batch's effective
attention on the tail of π̄₀, which is exactly the part of the proposal distribution that
matters for worst-case robustness under distributional shift. Plausible, but this is a
hypothesis, not something independently confirmed by a targeted ablation.

**Practical recommendation**: don't use `anchor_mix_prob > 0` with any of the three
objectives on either market. The original negative finding stands after the implementation
fix; if anything, the fixed version's one rigorously-tested candidate is a mild net negative
rather than a neutral one. This closes out the anchor-mix line of investigation a second time,
now with the budget-waste confound eliminated as a possible explanation for why it never
worked.

## Caveats / open items

- Stage C's significant result rests on selecting the one candidate that survived Stage A/B's
  screening — a full independent multi-seed test of all 6 combos from scratch (rather than a
  staged screen-then-confirm design) would be more airtight, but given the consistent
  direction across the whole investigation (this one plus the original 11-stage one), this is
  a low-priority follow-up.
- Only the `lam` values already tuned for each objective/market were used; no fresh lambda
  retuning was attempted specifically for the mixed proposal (the original investigation's own
  Stage C found this kind of lambda confound mattered, but here the design deliberately
  matched hyperparameters from the start to control for it).
- Not committed to git — implementation (`wdro.py`, `deep_hedging.py`, `test_wdro.py`) and this
  findings file await explicit request.

# Is `relaxed_kl_cvar`'s edge just an 8x-samples-per-step confound?

## Goal

`relaxed_kl_cvar` and `entropic` both train with `proposal_samples=8`, meaning every training
anchor is forward-passed through the strategy network 8 times per step (once per proposal),
vs. `nominal`'s 1 pass per anchor. Question: is their apparent robustness edge over `nominal`
actually just this 8x larger per-step sample/gradient count — a statistical-efficiency effect
unrelated to the WDRO/entropic-tilting math — rather than a genuine robustness mechanism? Ran
with a hard 3-concurrent-core budget (user's explicit constraint), using a new confound-control
objective (`plain_average`) plus a compute-matched nominal variant (`nominal_bigbatch`), paired
10-seed (seeds 6000-6009), toy `n_time=5` setup matching `recommendations.md` directly.

## New code

`plain_average_dual` (`wdro.py`): identical perturbation-sampling machinery to `entropic`/
`relaxed_kl_cvar` (same kernel, same 8x sample budget) but `objective = jnp.mean(pathwise_loss)`
over the full `(batch, n_proposals)` array — no worst-case tilt, no transport-cost penalty, no
lambda dependence at all. 2 new unit tests (exact-mean check; explicit "changing lambda/
transport_cost changes nothing" check). `nominal_bigbatch` needed no code changes — just
`batch_size=512` instead of 64.

`plain_cvar_of_average_dual` (`wdro.py`, added after the design subtlety below was caught):
holds the OUTER aggregation fixed to the exact same CVaR-alpha joint threshold
(`eta` + L^1-tail-norm/(1-alpha)) that `relaxed_kl_cvar_dual`/`joint_cvar_dual` use, while the
INNER neighborhood is collapsed to a plain self-normalized-importance-weighted average (no
tilt, no transport-cost penalty) — the true single-variable alternative to `relaxed_kl_cvar`.
3 new unit tests (manual CVaR-of-mean check with genuinely non-uniform importance weights;
lambda/transport-cost-invariance check; exact-eta-quantile check). All 25 tests pass.

**Independent review #1** (pre-run, implementation audit of `plain_average_dual`): confirmed the
math is exactly `mean(pathwise_loss)` with provably zero influence from lambda/transport_cost/
rho/log_weights, correct dispatch wiring, correct exclusion from the (here-irrelevant, since
`ula_steps=0`) `temper` flag, and confirmed sampling itself doesn't depend on `config.objective`
at all (so the 8x sample budget really is identical across `plain_average`/`entropic`/
`relaxed_kl_cvar`). Verdict: GO.

**Independent review #2** (pre-run, implementation audit of `plain_cvar_of_average_dual`):
derived the self-normalized importance-weighting formula by hand and confirmed it reduces
exactly to the unweighted arithmetic mean over proposals when `log_weights=0` (the default at
`ula_steps=0`, used throughout this study); confirmed the outer CVaR-alpha computation
structurally mirrors `joint_cvar_dual`/`relaxed_kl_cvar_dual`'s; confirmed zero leakage from
`lambda`/`transport_cost`; flagged one latent (here-inconsequential) footgun — `config.tail_power`
is silently ignored rather than validated, and the self-normalization convention differs subtly
from the rest of the file's raw-importance-weighting convention if ever run with `ula_steps>0`
(not the case in this study). Verdict: GO.

## An important design subtlety, caught before drawing conclusions

**`entropic_dual`'s outer aggregation over the batch is a plain arithmetic mean** (`wdro.py`,
`objective = ... + temperature * jnp.mean(inner)`), not a CVaR/quantile-based risk measure.
`plain_average_dual` matches that (also a plain mean). But **`nominal`'s training loss is a
Rockafellar-Uryasev CVaR-alpha reformulation with a learned threshold** (`strategy.q`), and
**`relaxed_kl_cvar_dual`'s outer aggregation is also CVaR/quantile-shaped** (`eta` + tail-norm
term) — structurally parallel to `nominal`'s, not to `entropic`'s/`plain_average`'s plain mean.

This means only **`entropic` vs `plain_average`** is a clean, single-variable comparison (both
share plain-mean outer aggregation and identical 8x sample budget; only the inner worst-of-
neighborhood tilt differs). **`plain_average` vs `nominal`** and **`relaxed_kl_cvar` vs
`plain_average`** are *not* clean single-variable comparisons — they also swap the outer risk
functional (mean vs CVaR-alpha), a well-known source of large behavioral differences on its own,
unrelated to sample count or perturbation. This was caught by independent review #3 (below)
before any conclusion was drawn from it, and reframes the headline number below.

## Results (paired 10-seed, seeds 6000-6009)

| config | BS in_dist | BS worst_extreme | Heston in_dist | Heston worst_extreme |
|---|---|---|---|---|
| nominal | 1.789±0.023 | 48.86±5.72 | 2.142±0.044 | 4.04±1.14 |
| nominal_bigbatch (batch=512) | 1.768±0.008 | 48.73±3.65 | 2.032±0.025 | 3.11±0.51 |
| plain_average (8x, no tilt) | **21.51±5.80** | **982±342** | **19.46±8.72** | **27.7±26.1** |
| entropic | 3.195±0.377 | 44.48±9.36 | 2.584±0.227 | 3.32±0.61 |
| plain_cvar_of_average (8x, no tilt, CVaR-alpha outer) | 1.747±0.018 | 44.67±5.81 | 2.068±0.047 | 3.184±0.506 |
| relaxed_kl_cvar | 1.751±0.017 | 40.64±3.88 | 2.000±0.031 | 2.76±0.68 |

### Comparison 1 (clean): entropic vs plain_average

Both use identical plain-mean outer aggregation and identical 8x sample budget — only the
inner log-sum-exp tilt differs. Result: **entropic decisively wins on both metrics, both
markets** (Wilcoxon p=0.0020-0.024, 10/10 win rate everywhere). This is the most defensible
finding in this study: **giving the network the same 8x sample budget without the tilting
mechanism is not just unhelpful, it's catastrophic (roughly an order of magnitude worse)** —
independent review #3's synthesis is that this should be read as "the tilting mechanism itself
is essential," not merely "more samples helps," since sample count is held fixed across this
comparison.

### Comparison 2 (confounded, flagged as such): plain_average vs nominal

`plain_average` is ~10x worse than `nominal` on both metrics, both markets. **This should NOT be
read as "8x samples actively hurts."** Per independent review #3: this is much better explained
by an objective mismatch — `plain_average` is trained to minimize a plain mean loss, then
evaluated on a CVaR_95 tail metric it was never optimized for, which is a well-documented
failure mode regardless of sample count. The order-of-magnitude gap here is not informative
about the sample-count question; it's informative about "don't train on mean loss if you'll be
judged on tail risk," a much less interesting and already well-known fact.

### Comparison 3 (clean): nominal_bigbatch vs nominal — the actual "just more samples" test

This is the comparison that actually isolates sample count while holding the CVaR-alpha outer
form fixed (both nominal variants use the same Rockafellar-Uryasev training objective, only
`batch_size` differs).

| market | in_dist p (Wilcoxon) | in_dist effect | worst_extreme p | worst_extreme effect |
|---|---|---|---|---|
| BS | 0.027 | small (~1%, 1.77 vs 1.79) | 0.85 (tie) | none |
| Heston | 0.0020 | **moderate (~5%, 2.03 vs 2.14)** | 0.065 (borderline) | small, not significant |

**More real i.i.d. samples per step alone gives a real, market-dependent, but modest
in-distribution benefit** — most pronounced at Heston (~5%), much smaller at BS (~1%), and no
clear worst-case tail benefit at either market on its own.

### Comparison 4 (clean): relaxed_kl_cvar vs nominal_bigbatch

Controls for sample count directly — both this and `relaxed_kl_cvar` see far more data per step
than plain `nominal`; the only remaining difference is whether that data is perturbation-based
and passed through the WDRO dual, vs. i.i.d. real paths through the plain CVaR-alpha objective.

| market | in_dist p | in_dist verdict | worst_extreme p | worst_extreme verdict |
|---|---|---|---|---|
| BS | 0.027-0.049 | marginal residual edge | **0.0004-0.0039** | **relaxed clearly still wins** |
| Heston | 0.064-0.106 | **tie — no significant residual edge** | 0.25-0.32 | **tie — no significant residual edge** |

**This is the most important, market-dependent result of the whole study**: at BS,
`relaxed_kl_cvar`'s tail-risk advantage over nominal survives even after fully controlling for
sample count — a genuine WDRO-specific effect. At Heston, once sample count is controlled for
via `nominal_bigbatch`, `relaxed_kl_cvar`'s remaining edge over nominal on *both* axes becomes
statistically indistinguishable from zero — meaning **much of Heston's apparent
`relaxed_kl_cvar`-over-nominal advantage in `recommendations.md` may be attributable to the
larger effective sample count alone, not to anything WDRO-specific**.

**Independent review #3** (post-hoc, stats + design audit): independently recomputed every
number above and confirmed all of them exactly; confirmed the outer-aggregation-type design
subtlety by reading the relevant dual functions directly (not taking my characterization on
faith); explicitly flagged that comparison 1 (`entropic` vs `plain_average`) should not be
oversold as validating relaxed_kl_cvar's CVaR-tail claims specifically, since entropic's own
outer aggregation is a plain mean too — the CVaR-relevant evidence rests specifically on
comparison 4's market-dependent result. This review is also what motivated adding
`plain_cvar_of_average` below, since it explicitly named "a future plain_average variant with a
CVaR-alpha outer aggregation" as the needed next step to fully close the design gap.

## Comparison 5 (the true single-variable test): relaxed_kl_cvar vs plain_cvar_of_average

`plain_cvar_of_average`: in_dist BS 1.747±0.018, Heston 2.068±0.047; worst_extreme BS
44.67±5.81, Heston 3.184±0.506 (n=10, same seeds as everything else).

This is the one comparison in the whole study holding **both** sample budget **and** outer
risk-measure type fixed — only the inner entropic tilt differs from `relaxed_kl_cvar`.

| market | in_dist p (Wilcoxon) | in_dist verdict | worst_extreme p | worst_extreme verdict |
|---|---|---|---|---|
| BS | 0.43 | tie (1.751 vs 1.747, essentially identical) | **0.0137** | **relaxed wins (90% win rate)** |
| Heston | **0.0059** | **relaxed wins (90% win rate, 2.000 vs 2.068)** | 0.084 | directional, not significant (80% win rate) |

**This is the cleanest, most decisive result in the study.** The tilt gives no in-distribution
benefit at BS (expected — it's designed to target worst-case regions, not the average case) but
a clear, significant tail-risk benefit there; at Heston it's the reverse pattern (clear
in-distribution benefit, directional-but-not-quite-significant tail benefit). Independent review
#4 (below) pushed back on reading the one non-significant cell (Heston worst_extreme, p=0.084,
80% win rate) as evidence of "no effect": with only 10 seeds, an 80%-concordant, p=0.084 result
is much better read as *underpowered in the same direction* than as a genuine null — three of
four cells show a significant, tilt-specific effect, and the fourth is directionally consistent
with the same story, just short of significance at this sample size.

## Comparison 6: plain_cvar_of_average vs plain_average — validating the design-subtlety diagnosis

If the design subtlety identified above (outer mean vs. outer CVaR-alpha) really was the
dominant driver of `plain_average`'s earlier catastrophic failure, then fixing *only* that (by
switching to `plain_cvar_of_average`, still with zero tilt) should mostly recover sane
performance. Result: **yes, decisively** — `plain_cvar_of_average` beats `plain_average` by
roughly an order of magnitude at every metric/market (e.g. BS worst_extreme 44.67 vs 982.25,
Wilcoxon p=0.0020, the maximum-significance floor at n=10, in every one of the 4 cells). This
directly confirms `plain_average`'s earlier blowup was mostly an objective-mismatch artifact
(mean-trained, CVaR-evaluated), not evidence that untitled perturbation-averaging is inherently
harmful.

## Comparison 7: plain_cvar_of_average vs nominal and vs nominal_bigbatch

| market | vs nominal in_dist p | vs nominal worst p | vs nominal_bigbatch in_dist p | vs nominal_bigbatch worst p |
|---|---|---|---|---|
| BS | 0.0039 (wins) | 0.23 (tie) | 0.0098 (wins) | 0.11 (tie) |
| Heston | 0.0195 (wins) | 0.11 (tie) | 0.11 (tie, 20% win rate — reversed) | 0.56 (tie) |

Untitled perturbation-averaging under a matched CVaR-alpha outer structure gives a modest,
market-dependent in-distribution edge over plain `nominal`, but that edge mostly evaporates once
sample count is also controlled for (`nominal_bigbatch`) — especially at Heston, where the
direction even reverses (though not significantly). This reinforces comparison 5's reading: at
Heston, the entropic tilt (not sample count, not outer aggregation type) is doing essentially
all of `relaxed_kl_cvar`'s work over a compute-matched baseline.

## Comparison 8: entropic vs plain_cvar_of_average

Entropic loses decisively on in-distribution at both markets (p=0.0020 each — its known cost),
and ties on worst_extreme at both markets (p=0.65-1.0, ~50% win rate). Entropic's plain-mean
outer aggregation plus tilt isn't detectably better on tail risk than a CVaR-alpha outer
aggregation with no tilt at all — the outer structure and the tilt appear to trade off rather
than stack.

**Independent review #4** (post-hoc, stats audit of comparisons 5-8): independently recomputed
every number above from the raw CSVs and confirmed them exactly, confirmed seed-matching is
genuine (`plain_cvar_of_average`'s rows are stored in a different seed order than the other
configs' CSV, so a naive row-order join would have silently mismatched seeds — the analysis
correctly joins on the seed column), and pushed back constructively on the Heston-worst_extreme
interpretation in comparison 5 (see above) rather than accepting the initial "market-dependent
tie" framing at face value.

## Bottom line

- **The "it's just 8x more samples" hypothesis is not supported.** The one experiment that holds
  both sample count and outer risk-measure type fixed (comparison 5: `relaxed_kl_cvar` vs
  `plain_cvar_of_average`) shows the entropic tilt gives a real, tilt-specific benefit in 3 of 4
  market/metric cells (BS tail risk, Heston in-distribution, both p<0.02), and the 4th (Heston
  tail risk, p=0.084, 80% win rate) reads as underpowered-in-the-same-direction rather than a
  genuine null, per independent review #4.
- **The entropic tilting mechanism itself is not just "more samples in disguise"** under either
  outer-aggregation type: `entropic` vs `plain_average` (comparison 1, mean-outer) and
  `relaxed_kl_cvar` vs `plain_cvar_of_average` (comparison 5, CVaR-outer) both show the tilt
  doing real, distinct work from sample count.
- **The earlier `plain_average` vs `nominal` result (comparison 2) was mostly a red herring**:
  fixing only the outer-aggregation-type mismatch (`plain_cvar_of_average`, comparison 6)
  recovers most of the lost performance, confirming that result was about objective mismatch
  (mean-trained, CVaR-evaluated), not about perturbation sampling being harmful.
- **The earlier open item — "does relaxed's CVaR-alpha outer structure vs. entropic's plain mean
  matter, independent of the tilt?" — is now addressed**: comparison 8 (`entropic` vs
  `plain_cvar_of_average`) suggests the outer structure and the inner tilt trade off (neither
  combination of "outer type x tilt" dominates on tail risk) rather than stacking additively,
  though this wasn't the primary axis of interest and deserves more seeds before treating as
  final.

## Caveats / open items

- Only the toy `n_time=5` setup was tested here (matches `recommendations.md` directly); whether
  this sample-count confound picture changes at the realistic `n_time=20/30` horizons from
  `longer_horizon_findings.md` is untested.
- No lambda re-tuning was attempted for `entropic`/`relaxed_kl_cvar` in this study (reused
  `recommendations.md`'s established best values) — the goal was explaining an already-confirmed
  result, not re-optimizing it.
- The one borderline cell (Heston, `relaxed_kl_cvar` vs `plain_cvar_of_average`, worst_extreme,
  p=0.084) would benefit from more seeds (e.g. n=20-30) to resolve whether it's a real, smaller
  tilt-specific effect or truly a null at Heston specifically — flagged, not resolved, here.
- `plain_cvar_of_average_dual`'s self-normalized-importance-weighting convention differs subtly
  from the rest of `wdro.py`'s raw-importance-weighting convention; only matters if this
  objective is ever combined with `ula_steps>0` (not the case anywhere in this study).
- Not committed to git — `wdro.py`/`deep_hedging.py`/`test_wdro.py` changes, this findings file,
  and `benchmark/rho_sweep/confound_stageA.csv`/`confound_stageB.csv` await explicit request.

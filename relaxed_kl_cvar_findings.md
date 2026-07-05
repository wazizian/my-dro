# Relaxed-marginal KL+L^p WDRO ("mixed CVaR-WDRO") findings

## Goal

Implement and stress-test the combined formulation from `note_wdro_solve.typ`'s
`@relaxed-kl-primal`/`@relaxed-kl-dual` (lines 56-107): unlike pure entropic WDRO (which
smooths every training anchor uniformly via a per-anchor KL/log-sum-exp term) or pure
`joint_cvar`/`per_sample_cvar` (which apply a raw, un-smoothed CVaR threshold directly to the
pathwise loss), this formulation computes the entropic-smoothed per-anchor value h_lambda(xi)
first, then wraps *that* in an outer joint CVaR-style threshold (eta, alpha) across the batch
of anchors ~ P. It combines both regularizations: KL-smoothing within each anchor's proposal
neighborhood, CVaR-style selectivity across anchors. Task: show this "mixed" objective offers
a superior tradeoff vs pure entropic, on both BS and Heston, with extensive tuning and regular
independent review.

## Implementation

New objective `relaxed_kl_cvar` added to `RobustConfig.objective` and `relaxed_kl_cvar_dual`
in `wdro.py`, wired into `deep_hedging.py`'s dispatch and `temper` condition. Reuses
`entropic_dual`'s log-sum-exp `h_lambda` computation exactly, then reuses `joint_cvar_dual`'s
tail-norm pattern applied to `h_lambda` (shape `(batch,)`) instead of the raw penalized
pathwise loss (shape `(batch, n_proposals)`). 2 new unit tests added to `test_wdro.py`
(exact-value check against manual numpy recomputation; reduction-to-entropic sanity check when
eta is forced below every h_lambda so the whole batch falls in the tail). All 18 tests pass.

**Independent review #1** (implementation audit, before any sweeping): confirmed math and
wiring correct against the note; no bugs found; one cosmetic-only inconsistency (an eta-gradient
masking list omits `relaxed_kl_cvar` when `exact_eta=True`, but this is inconsequential since
`state.eta` is structurally disconnected from the graph in that case anyway). Verdict: safe to
sweep.

## Stage A: coarse ranking (3-seed avg, seeds 2000-2002)

Swept `lambda` x `alpha` x `kl_epsilon` on both markets, `tail_power=1`, `exact_eta=True`,
`cost_power=1`, gaussian, `proposal_scale=0.15`. Best BS point (lam=0.05, alpha=0.9, eps=0.03):
in_dist=1.76, worst_extreme=36.2 — already visibly close to nominal on in_dist while much
better than nominal's ~48-51 on worst_extreme. Best Heston point in the same neighborhood.

## Stage B: matched-lambda pure-entropic baseline (3-seed, same seeds 2000-2002)

Ran pure entropic at the identical (market, lambda, kl_epsilon) grid as Stage A, to isolate
whether the outer CVaR wrap helps beyond what the same lambda already buys entropic (the
anchor-mix investigation's central methodological lesson: always compare at matched
hyperparameters). Found: at every matched lambda, `relaxed_kl_cvar`'s in_dist stayed near
nominal (~1.75) while entropic's climbed to ~3.2-3.5 — a large, consistent gap unrelated to
lambda choice.

## Stage C: single-seed wide-lambda frontier, BS (seed 2100)

Widened lambda range far beyond Stage A/B (entropic: 0.0005-0.1; relaxed: 0.02-3.0) to map
each objective's full frontier rather than one point. Found a striking structural pattern:
entropic's in_dist floor never dropped below ~3.36 even at its smallest lambda (0.0005),
while `relaxed_kl_cvar`'s in_dist stayed pinned at ~1.73-1.88 across its *entire* lambda range
(0.02 to 3.0) — i.e., in_dist is nearly insensitive to lambda for the new objective. Mechanistic
explanation: the outer `relu(h_lambda - eta)` tail term only back-propagates gradient through
the worst `(1-alpha)` fraction of anchors in each training batch; the other ~90% of anchors see
no robustness pressure at all, so the resulting strategy tracks near-nominal in-distribution
behavior by construction, unlike entropic which perturbs every anchor's target uniformly via
the proposal-ball log-sum-exp (a floor that persists even as lambda -> 0, since kl_epsilon > 0
alone induces a worst-of-neighborhood-ball effect regardless of the transport-cost penalty).

Single-seed data was noisy (entropic's in_dist was non-monotonic in lambda), so this stage's
numbers were provisional pending multi-seed confirmation (Stage D).

## Stage D: 3-seed frontier confirmation + matched nominal baseline (seeds 2200-2202)

Confirmed Stage C's structural pattern survives averaging: entropic's in_dist stayed in
2.86-3.13 (BS) / 2.57-2.85 (Heston) across all tested lambdas; relaxed's stayed at 1.74-1.77
(BS) / 2.00-2.11 (Heston), i.e. close to the matched nominal baseline (BS 1.79/51.1, Heston
2.18/4.01, run on the identical seeds/eval grid in Stage E since no pre-existing nominal
baseline used this exact harsh stress grid).

Cross-referencing against nominal:
- **BS**: relaxed's best point (lam=0.1: in=1.755, worst=45.8) beats nominal (1.79/51.1) on
  both axes but does *not* reach entropic's best worst_extreme (35.7-40.3, at a much higher
  in-dist cost of 2.86-3.13) — a genuinely different frontier point, not a domination.
- **Heston**: relaxed's best point (lam=0.02: in=2.00, worst=2.57) beat *both* nominal
  (2.18/4.01) *and* entropic's apparent best 3-seed point (lam=0.05: 2.57/3.12) on both axes —
  the strongest possible claim, flagged for rigorous paired confirmation.

## Stage F: paired 10-seed confirmation (fresh seeds 2300-2309)

- Heston, relaxed (lam=0.02) vs nominal: in_dist p<1e-4 (100% win), worst_extreme p=0.0093 /
  Wilcoxon p=0.002 (100% win). **Confirmed.**
- Heston, relaxed (lam=0.02) vs entropic (lam=0.05): in_dist p<1e-4 (100% win), worst_extreme
  p=0.0012 / Wilcoxon p=0.0039 (90% win). Looked like full dominance over entropic — see
  Stage G correction below.
- BS, relaxed (lam=0.1) vs nominal: in_dist p<1e-4 (100% win), worst_extreme p=0.0008 /
  Wilcoxon p=0.0039 (90% win). **Confirmed** — a genuine, if modest, BS win over nominal.

**Independent review #2** (statistics + confound audit): recomputed all p-values/win-rates
independently from the raw CSV and confirmed them exactly. Outlier check (removing Heston
nominal's one high seed, worst_extreme=10.1) *tightened* significance rather than weakening it
— not an outlier artifact. Flagged one legitimate open concern: entropic's Heston lambda grid
in Stage D was only 5 points at 3 seeds, and its worst_extreme trend across that grid was
non-monotonic — "entropic's best" (lam=0.05) was plausible but not confirmed as truly optimal.
Recommended a finer paired sweep before calling the "beats entropic on both axes" claim final.

## Stage G: finer entropic lambda grid, paired 10-seed (same seeds 2300-2309)

Swept entropic at lam in {0.02, 0.03, 0.04, 0.06, 0.08} on Heston, same seeds as Stage F.
Found a better entropic point than Stage D's coarse grid suggested: **lam=0.04, in_dist=2.48,
worst_extreme=2.94** (vs Stage D's lam=0.05 point of 2.57/3.12). Re-ran the paired comparison
against `relaxed_lam0.02` on this matched-seed, better-tuned entropic point:

| metric | entropic (lam=0.04) mean | relaxed (lam=0.02) mean | win rate (relaxed) | ttest p | wilcoxon p |
|---|---|---|---|---|---|
| in_distribution | 2.484 | 2.005 | 100% | 1.5e-8 | 0.0020 |
| worst_extreme | 2.940 | 2.688 | 70% | 0.185 | 0.160 |

**Correction to Stage F's headline claim**: against this more carefully tuned entropic point,
the worst_extreme advantage for `relaxed_kl_cvar` is no longer statistically significant
(direction still favors relaxed, 70% win rate, but p=0.16-0.19 at n=10) — this is *not* full
Pareto dominance on both axes. The in-distribution advantage remains large and overwhelmingly
significant (p=1.5e-8) regardless of which entropic lambda is used as the comparison point.

## Bottom line

**The honest, reviewer-verified result**: `relaxed_kl_cvar` does not reduce Heston's
worst-case tail risk below what a well-tuned pure entropic objective already achieves — that
comparison is a statistical tie (p=0.16-0.19, matched seeds). What it *does* deliver, robustly
and with large effect size on both markets:

- **A materially better in-distribution / tail tradeoff shape than pure entropic**: it reaches
  entropic's own best achievable worst-case tail protection (statistically indistinguishable,
  70% directional win rate) while paying almost none of entropic's in-distribution cost
  (Heston: in_dist 2.00 vs entropic's 2.48, p=1.5e-8; BS: in_dist 1.75 vs nominal's 1.79,
  essentially free). Entropic's in-distribution cost is structural — it never dropped below
  ~2.86 (BS) / ~2.48 (Heston) across every lambda tested in Stages C/D/G, because
  its per-anchor log-sum-exp smoothing pushes every training sample toward its local worst
  case regardless of the transport-cost weight. `relaxed_kl_cvar`'s outer CVaR wrap instead
  concentrates that pressure on only the worst `(1-alpha)` fraction of anchors per batch,
  leaving the rest to train close to nominal.
- **A clear, significant improvement over doing nothing (nominal)** on both markets: Heston
  worst_extreme 2.57 vs nominal's 4.01 (p=0.0093/0.002), BS worst_extreme 45.8-48.7 vs
  nominal's 51.1 (p=0.0008/0.0039), both at near-zero in-distribution cost.

So: not "beats pure entropic outright," but "gets entropic's own best tail-risk protection for
close to nominal's in-distribution cost" — a genuinely different and arguably more practically
useful point on the tradeoff curve, not a strict dominance. 7 stages, 2 independent reviewer
passes (implementation + statistics), ~360 training runs, fresh non-overlapping seed ranges at
every confirmation stage.

## Caveats / open items

- Only `cost_power=1`, gaussian proposal, `proposal_scale=0.15`, `tail_power=1` were tuned in
  depth; laplace/`cost_power=2`/wider scale were not swept for this objective (the earlier
  anchor-mix investigation found family/cost_power interactions were often market-specific and
  not always intuitive — worth checking here too if this line of work continues).
- BS's relaxed-vs-entropic comparison was not run at a matched, finely-tuned entropic lambda
  the way Heston's was (Stage G) — the BS "different frontier point, not a domination" framing
  rests on Stage D/E's coarser grid.
- Not committed to git — implementation (`wdro.py`, `deep_hedging.py`, `test_wdro.py`) and this
  findings file await explicit request.

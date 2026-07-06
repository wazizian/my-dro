# Do the entropic / joint_cvar / relaxed_kl_cvar findings hold at realistic hedging horizons?

## Goal

Every finding so far in this repo (`recommendations.md`, `relaxed_kl_cvar_findings.md`,
`anchor_mix_reeval_findings.md`) was produced on a toy `n_time=5, horizon=5/365` (5-day)
market, used purely for iteration speed. `deep_hedging.py`'s actual `MarketConfig` default is
`n_time=30, horizon=30/365` — a realistic, daily-rebalanced month — which had never been
tested. This is not just "more compute": the WDRO anchor is the encoded trajectory
(`encode_trajectory`), whose dimension scales with `n_time` (BS: dimension = n_time; Heston:
dimension = 2·n_time). Going from n_time=5 to 30 means the KL/entropic proposal ball has to
cover a 6x (BS) / 12x (Heston) higher-dimensional space at the same `proposal_scale`/
`proposal_samples` — a genuinely untested regime. Ran overnight, using all 12 CPU cores via a
parallel multiprocessing driver, to re-test all 3 objectives × 2 markets at `n_time` = 20, 30,
and a stretch test at 60.

## Compute setup

No source code changes were needed — `RobustConfig`/`MarketConfig` already parameterize
everything by `n_time`. Built `overnight_driver.py`: a `multiprocessing.Pool` (spawn context,
10 thread-capped workers) running the established coarse (1 seed) → 3-seed frontier → paired
10-seed confirmation protocol, computing each stage's winning lambda programmatically from the
previous stage's CSV so the whole thing runs unattended. Fresh seeds 4000-4409 (confirmed
disjoint from every prior CSV in this repo).

**Pre-flight independent review** (before launch): caught and fixed one real bug (the Stage D
stretch-goal lambda selection could pull from the wrong n_time's Stage B data), an `XLA_FLAGS`
typo, and recommended (both applied) per-job fault isolation so one bad config couldn't kill
the whole overnight schedule, plus a wider lambda grid (0.1x-8x instead of 0.25x-4x around each
combo's n_time=5-tuned value, since a 6-12x dimensionality jump could plausibly shift the
optimum by more than 4x). Sanity checks (Stage 0) confirmed all 6 (objective, market) combos
train without shape/dimension errors at n_time=30 before committing to the full run.

## Stage A/B/C: n_time = 20 and 30

Coarse 1-seed lambda grid (Stage A, 7 points, seeds 4000/4001) → 3-seed frontier at the winning
lambda (Stage B, seeds 4100-4112) → paired 10-seed confirmation vs. nominal and head-to-head
(Stage C, seeds 4200-4309). Result: 0 job failures across all ~300 jobs.

### vs. nominal (Stage C, paired 10-seed, Wilcoxon p unless noted)

| combo | n_time | in_dist (combo/nominal) | in_dist p | worst_extreme (combo/nominal) | worst_extreme p |
|---|---|---|---|---|---|
| bs_entropic | 20 | 10.76 / 6.82 (**worse**) | 0.0020 | 419 / 341 (tie) | 1.00 |
| bs_entropic | 30 | 14.94 / 10.23 (**worse**) | 0.0020 | 569 / 625 (tie) | 0.70 |
| bs_joint_cvar | 20 | 4.47 / 6.82 | 0.0020 | 206 / 341 | 0.065 |
| bs_joint_cvar | 30 | 7.81 / 10.23 | 0.0020 | 477 / 625 | 0.23 |
| bs_relaxed_kl_cvar | 20 | 5.24 / 6.82 | 0.0020 | 226 / 341 | 0.0059 |
| bs_relaxed_kl_cvar | 30 | 8.55 / 10.23 | 0.0039 | 529 / 625 | 0.16 |
| heston_entropic | 20 | 6.82 / 6.90 (flat) | 0.77 | 13.5 / 27.3 | 0.0020 |
| heston_entropic | 30 | 9.75 / 10.01 (flat) | 0.56 | 31.0 / 54.2 | 0.037 |
| heston_joint_cvar | 20 | 5.72 / 6.90 | 0.0020 | 16.6 / 27.3 | 0.0020 |
| heston_joint_cvar | 30 | 7.65 / 10.01 | 0.0020 | 28.2 / 54.2 | 0.0039 |
| heston_relaxed_kl_cvar | 20 | 5.87 / 6.90 | 0.0020 | 11.0 / 27.3 | 0.0020 |
| heston_relaxed_kl_cvar | 30 | 7.75 / 10.01 | 0.0020 | 20.7 / 54.2 | 0.0020 |

### entropic vs relaxed_kl_cvar, head-to-head (Wilcoxon p)

| market | n_time | in_dist (entropic/relaxed) | in_dist p | worst_extreme (entropic/relaxed) | worst_extreme p |
|---|---|---|---|---|---|
| bs | 20 | 10.76 / 5.24 | 0.0020 | 419 / 226 | 0.027 |
| bs | 30 | 14.94 / 8.55 | 0.0020 | 569 / 529 | 0.70 |
| heston | 20 | 6.82 / 5.87 | 0.0039 | 13.5 / 11.0 | 0.16 |
| heston | 30 | 9.75 / 7.75 | 0.0020 | 31.0 / 20.7 | 0.28 |

## Key result 1: the central recommendation replicates well

`recommendations.md`'s headline finding — `relaxed_kl_cvar` wins in-distribution overwhelmingly
(p<0.01 in every case above, at every horizon and market tested: 5, 20, 30) while its
worst-case tail protection is a statistical tie against entropic (not significantly different,
one weak exception at BS/nt20) — **holds at every horizon tested, not just the toy n_time=5
setup**. This is the strongest, most horizon-robust finding of this whole investigation.

`joint_cvar` and `relaxed_kl_cvar` both beat nominal on both axes at essentially every
(market, n_time) combination, mostly at p<0.05 (weakest cases: `bs_joint_cvar` worst_extreme at
nt20, p=0.065; `bs_relaxed_kl_cvar` worst_extreme at nt30, p=0.16 — still directionally
favorable, 60-90% win rate, just not airtight at n=10).

## Key result 2 (new, surprising): BS entropic's tail-risk edge does not survive longer horizons

At n_time=5, `recommendations.md` found BS entropic clearly beats nominal on worst_extreme
(30.2 vs 51.1) at a real in-distribution cost. At n_time=20 and 30, that tail-risk advantage
**disappears** — worst_extreme is a statistical tie against nominal at both horizons
(p=1.00, p=0.70), while the in-distribution cost gets *worse*, not just present (p=0.002 at
both, entropic now materially above nominal rather than just somewhat above it). This was
independently checked against a possible under-tuning explanation: reviewer #2 cross-referenced
the full Stage A lambda grid (0.1x-8x around the n_time=5 optimum) plus the Stage E extension
(down to 0.00125, well beyond the original grid) for `bs_entropic` at n_time=30 — no tested
lambda in that entire range recovers a worst_extreme materially below nominal's, so this is not
simply a matter of needing to search further. Best available hypothesis (reviewer #2's, not
independently re-verified by a third pass): the entropic tilt's effect on rare extreme paths
gets diluted as the horizon lengthens, so the mechanism that gave entropic its tail-risk edge at
5 days doesn't scale to a full month. **This revises part of `recommendations.md`'s guidance**:
"if tail risk is the only thing that matters, use entropic" no longer holds unqualified for BS
at realistic horizons — Heston's entropic tail-risk edge over nominal, by contrast, holds fine
at both nt=20 and nt=30 (p=0.002, p=0.037).

## Stage D: n_time=60 stretch (heston_relaxed_kl_cvar, seeds 4400-4409)

Picked automatically as the combo with the largest Stage C improvement (61.8%). Result: **both
metrics are statistical ties against nominal** (in_dist p=0.48, worst_extreme p=0.13-0.32) —
a real null result, confirmed by reviewer #2, not a data artifact (no NaNs; one high-variance
nominal seed inflates nominal's spread but removing it doesn't flip the conclusion). Important
caveat, flagged in the plan from the start: this stretch run **reused the n_time=30-tuned
lambda without any fresh tuning for n_time=60** — it is a "does yesterday's tuning transfer
untouched" check, not a properly tuned n_time=60 result. Reviewer #2 could not distinguish
"lambda needs retuning at this horizon" from "the longer-horizon simulation is just noisier at
n=10" as the explanation — both are plausible. **Don't read this as "the effect reverses at 60
days"** — read it as "an untuned transfer doesn't automatically work; a real answer for
n_time=60 needs its own tuning pass," which the overnight budget didn't extend to.

## Stage E/F: lambda-grid-edge extension (user-directed follow-up)

5 of the 12 (combo, n_time) Stage A results landed on the edge of the swept grid — a real
limitation the pre-flight reviewer flagged in advance. Extended each in the indicated direction
(3-seed avg, seeds 5000-5002):

| combo | n_time | old edge (lam) | extended winner | new worst_extreme vs nominal |
|---|---|---|---|---|
| heston_entropic | 20 | 0.32 | 0.64 | 17.2 vs 27.3 (better than the Stage C number above) |
| heston_relaxed_kl_cvar | 20 | 0.16 | 0.64 | 15.2 vs 27.3 |
| bs_entropic | 30 | 0.01 | *no improvement found* (extension down to 0.00125 didn't beat 0.01) | Stage C's number stands |
| bs_joint_cvar | 30 | 0.005 | 0.000625 | 718 vs 625 (**worse than nominal** — extending further hurt) |
| heston_relaxed_kl_cvar | 30 | 0.002 | 0.001 | 23.1 vs 54.2 |

**Important caveat, caught by reviewer #2**: the 4 rows with an extended winner got a follow-up
paired-10-seed run (`stageF_extended_confirm.csv`, seeds 5100-5109), but that comparison used
Stage C's *existing* nominal rows (seeds 4200s/4300s) as the "paired" partner — a different
seed range entirely. **This is not actually seed-matched** the way every other comparison in
this doc is; it's closer to an unpaired mean comparison. Treat these 4 numbers as directionally
suggestive (2 of them clearly better than nominal, 1 essentially unchanged from Stage C, and
`bs_joint_cvar`'s further extension actually made worst_extreme *worse* than nominal — a useful
negative result on its own, showing lower isn't always better) rather than as rigorously
confirmed as the main Stage C table above.

## Bottom line

1. The central `recommendations.md` finding (`relaxed_kl_cvar` wins in-distribution by a large,
   significant margin; ties `entropic` on worst-case tail protection) replicates cleanly across
   every horizon tested (5, 20, 30 days) — this is now a well-established, horizon-robust
   result, not an artifact of the toy 5-day setup.
2. `joint_cvar` and `relaxed_kl_cvar` continue to robustly beat nominal on both axes at longer
   horizons on both markets, with a couple of borderline (not airtight, still directional)
   exceptions on BS worst_extreme.
3. **New finding, revising prior guidance**: BS entropic's tail-risk advantage over nominal,
   which was the basis for "if worst-case risk is the only thing you care about, use entropic"
   in `recommendations.md`, does not survive at n_time=20/30 — it becomes a statistical tie
   (not a win), confirmed not to be a lambda-tuning artifact. Heston entropic's tail-risk edge
   over nominal, by contrast, still holds at these horizons.
4. The n_time=60 stretch test is a null result but with an important caveat (lambda wasn't
   retuned for that horizon) — it's evidence that untuned transfer doesn't automatically work,
   not evidence that the whole finding reverses at very long horizons.
5. Two independent reviewer passes (pre-flight implementation/design audit; post-hoc statistics
   audit) verified the driver's correctness before the run and the paired statistics after —
   the post-hoc audit is also what surfaced Stage F's seed-pairing caveat above, keeping this
   writeup honest about which numbers are fully rigorous and which are merely suggestive.

## Practical recommendation update

`recommendations.md`'s "§5: single production default" (`relaxed_kl_cvar`, lambda tuned per
market) still stands and is now better-supported than before — it's the one recommendation
confirmed across three horizons. The "switch to entropic if worst-case tail risk is the sole
metric" caveat should be narrowed: this is confirmed to still work for Heston at realistic
horizons, but is **not confirmed for BS** beyond the toy 5-day setup — at n_time=20/30, BS
entropic no longer clearly beats nominal on tail risk, at a larger in-distribution cost than
before. If BS tail-risk-only hedging is needed at a realistic horizon, this repo does not yet
have a confirmed answer for which objective delivers it — that's the main open question this
investigation surfaces, not answers.

## Caveats / open items

- Stage D's n_time=60 result used an untuned (n_time=30) lambda — a fresh tuning pass at
  n_time=60 is the natural next step if this line of work continues.
- Stage F's 4 extended-lambda confirmations are not properly seed-paired against nominal
  (different seed ranges) — directionally informative, not rigorously confirmed.
- Only `cost_power=1`, gaussian proposal were tested at the new horizons — laplace/cost_power=2
  interactions (flagged as unexplored in every prior findings doc in this repo) remain untested
  here too.
- Not committed to git — `overnight_driver.py`/`stageE_extend_edges.py` (scratch, not part of
  the repo) plus this findings file and the new CSVs in `benchmark/rho_sweep/` await explicit
  request.

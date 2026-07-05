# Recommendations going forward

Based on `relaxed_kl_cvar_findings.md` (7 stages, ~360 runs, 2 independent reviewer passes)
and `anchor_mix_findings.md` (11 stages, ~500 runs, exhaustively negative).

## 1. The core decision: which objective depends on which question you're asking

**"Minimize worst-case risk at any cost" -> pure entropic still wins.**

| market | config | in_distribution | worst_extreme |
|---|---|---|---|
| Heston | entropic, best tuned (lam=0.04) | 2.48 | **2.94** |
| Heston | relaxed_kl_cvar (lam=0.02) | 2.00 | 2.69 |
| BS | entropic, best known (lam=0.1, gaussian) | 2.62 | **30.2** |
| BS | relaxed_kl_cvar (lam=0.1) | 1.75 | 45.8 |

On Heston, relaxed's worst_extreme (2.69) is directionally lower than entropic's (2.94), but
paired 10-seed testing (same seeds 2300-2309) puts this at p=0.185 (t-test) / p=0.160
(Wilcoxon), 70% win rate — **not statistically significant, a tie**. On BS, entropic's tuned
point (30.2) is clearly better than relaxed's (45.8) — here entropic outright wins on
worst-case, at the cost of nearly doubling in-distribution loss (2.62 vs nominal 1.79).
**If the objective function is "get the lowest possible worst_extreme, full stop," entropic is
the better choice on both markets** (matches or beats relaxed's tail protection every time).

**"Get meaningful robustness without wrecking nominal performance" -> relaxed_kl_cvar wins.**

| market | config | in_distribution | vs nominal | worst_extreme | vs nominal |
|---|---|---|---|---|---|
| Heston | nominal | 2.18 | — | 4.01 | — |
| Heston | entropic (lam=0.04) | 2.48 | +14% | 2.94 | -27% |
| Heston | relaxed_kl_cvar (lam=0.02) | 2.00 | **-8%** | 2.69 | **-33%** |
| BS | nominal | 1.79 | — | 51.1 | — |
| BS | entropic (lam=0.1) | 2.62 | +46% | 30.2 | -41% |
| BS | relaxed_kl_cvar (lam=0.1) | 1.75 | **-2%** | 45.8 | **-10%** |

Paired 10-seed confirmation (fresh seeds 2300-2309): relaxed_kl_cvar's in-distribution
advantage over entropic is large and highly significant on Heston (mean 2.00 vs 2.48,
p=1.5e-8, 100% win rate). On both markets relaxed beats nominal outright on both axes
(Heston: in_dist p<1e-4, worst_extreme p=0.0093; BS: in_dist p<1e-4, worst_extreme p=0.0008).
**If the objective is "buy some tail protection without paying much nominal-performance tax,"
relaxed_kl_cvar is the better choice** — it captures most or all of entropic's worst-case
improvement over nominal (Heston: -33% vs entropic's -27%; BS: only -10% vs entropic's -41%,
so the BS case is weaker) while costing almost nothing in-distribution (-2% to -8% vs
entropic's +14% to +46%).

**Bottom line on this framing**: the two objectives sit at genuinely different points on the
in-distribution/tail-risk tradeoff curve, not one strictly dominating the other. Pick entropic
if tail risk is the only thing that matters; pick relaxed_kl_cvar if in-distribution
performance matters too, especially on Heston where it gets tail protection at effectively no
cost.

## 2. Don't chase further worst-case tuning on relaxed_kl_cvar

Stage G showed the apparent "beats entropic on both axes" result from Stage F (p=0.0012 on
worst_extreme, 90% win rate) was an artifact of an under-tuned entropic comparison point
(lam=0.05, 3-seed coarse grid). Once entropic was tuned properly (lam=0.04, same 10 seeds),
the worst_extreme gap collapsed to a non-significant tie (p=0.16-0.19). Re-running more
lambda/alpha grids on relaxed_kl_cvar chasing a worst-case win over entropic would just
re-litigate a result already checked and found negative.

## 3. Untuned dimensions worth one more pass, if this continues

- Laplace proposal family and `cost_power=2` were never swept for `relaxed_kl_cvar` — the
  anchor-mix investigation found family/cost_power interactions are often market-specific, so
  this is a plausible place a real improvement could still be hiding.
- BS never got the same fine-lambda entropic comparison Heston got in Stage G — the BS "-10%
  vs entropic's -41%" framing above rests on Stages D/E's coarser lambda grid, so it's weaker
  evidence than the Heston numbers.

## 4. Skip further work on anchor-mix and joint/per-sample CVaR

Exhaustively closed out negative across 11 stages, ~500 runs (`anchor_mix_findings.md`): no
mean-tradeoff improvement over pure entropic/joint_cvar/per_sample_cvar on any objective x
market x proposal-family combination tried. One narrow, non-significant exception
(Heston+entropic+Laplace variance-reduction effect) is documented but not actionable without
new evidence. No new angle has been proposed here — don't reopen without one.

## 5. If a single production default is needed

`relaxed_kl_cvar`, alpha=0.9, `kl_epsilon=0.03`, `cost_power=1`, gaussian proposal, lambda
tuned per market (0.02 Heston / 0.1 BS) — the one config with reviewer-verified, paired-
significant evidence of beating nominal on both axes on both markets, at a materially lower
in-distribution cost than entropic. Only switch to entropic if worst-case tail risk is the sole
metric that matters, in which case accept the larger in-distribution hit that comes with it.

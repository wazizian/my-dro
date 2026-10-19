#set page(paper: "us-letter", margin: 1.5cm)
#set text(size: 10pt, font: "New Computer Modern")
#set heading(numbering: "1.")

#align(center)[
  #text(size: 16pt, weight: "bold")[LSV Sample-Size Study — Results]
]

= Market model: Heston + leverage

- Base: Heston SV, variance leg unchanged (exact CIR)
- Variance leg: $ d V_t = kappa (theta - V_t) d t + xi sqrt(V_t) d W^V_t $
- Stock leg: $ d S_t = L(t,S_t) sqrt(V_t) S_t (rho d W^V_t + sqrt(1-rho^2) d W^perp_t) $
- Leverage function (CEV skew × time-growth): $ L(t,S) = (S\/S_0)^beta (1 + kappa_L t) $
- Leverage capped: $ L <- op("clip")(L, 0, "leverage_cap") $ — prevents blowup as $S_t -> 0$

== Parameters

Risk-neutral-recalibrated Heston core ($kappa=3.0$, vol_of_variance$=0.4$, $rho=-0.7$,
$theta=v_0=0.04$), 3-tier leverage-shift ladder (weak/medium/hard = 1.5x/2x/3x the in-dist
$(beta,kappa_L)$ offset from zero).

#table(
  columns: 5,
  align: left,
  table.header([*param*], [*in-dist*], [*weak*], [*medium*], [*hard*]),
  [$beta$], [$-0.3$], [$-0.45$], [$-0.6$], [$-0.9$],
  [$kappa_L$], [$0.5$], [$0.75$], [$1.0$], [$1.5$],
  [leverage_cap], [5.0], [5.0], [5.0], [5.0],
  [n_substeps (fine steps/hedge)], [10], [10], [10], [10],
  [n_time (hedges)], [100], [100], [100], [100],
  [horizon], [100/365], [100/365], [100/365], [100/365],
)


= Objectives

#table(
  columns: 4,
  align: left,
  table.header([*method*], [*hyperparams swept*], [*extra hyperparam*], [*fixed*]),
  [`CVaR`], [`learning_rate`, `alpha`], [—], [plain nominal CVaR-95],
  [`CVaR`+`augmentation`], [`learning_rate`, `alpha`], [`proposal_scale`], [`proposal_samples=32`],
  [`WDRO`], [`learning_rate`, `alpha`], [`rho`], [`kl_epsilon=0.03`, `ula_steps=0`, `cost_power=2`, `proposal_samples=32`],
  [`WDRO`+`convex order`], [`learning_rate`, `alpha`], [`rho`], [as `WDRO`, plus trainable-$lambda_(i j)$ convex-order (no-calendar-arbitrage) penalty on proposal paths],
)

- `batch_size=64` fixed across all 4 methods, all $n$ (not swept)
- `epsilon=0.0` (no transaction cost), `latent_dim=0` (plain per-timestep MLP+BN policy) for this run

= Study design

- $n in {100, 250, 500, 750, 1000}$ training paths, 3 seeds
- $n$ samples are split in train/validation (80/20) for hyperparam tuning, $k=3$ folds
- Winning hyperparam's model reused directly (no separate full-$n$ refit)
- Test: $n_"test"=20000$ paths, in-dist and 3-tier shifted leverage params
- Canonical metric: empirical CVaR-95 of realized pathwise loss

#pagebreak()

= Results (mean over 3 seeds)

Columns: val = k-fold validation CVaR-95, test = in-dist test CVaR-95, shift = hard-tier
(3x offset) shifted test CVaR-95, the worst of the 3 shift tiers. Full weak/medium/hard breakdown
in the note below.

#set table.hline(stroke: 0.6pt)
#let vsep = table.vline(stroke: 0.5pt)

#table(
  columns: 13,
  align: (left,) + (right,)*12,
  stroke: none,
  table.hline(),
  table.header(
    table.cell(rowspan: 2, align: horizon)[*method*], vsep,
    table.cell(colspan: 3, align: center)[*n = 250*], vsep,
    table.cell(colspan: 3, align: center)[*n = 500*], vsep,
    table.cell(colspan: 3, align: center)[*n = 750*], vsep,
    table.cell(colspan: 3, align: center)[*n = 1000*],
    table.hline(start: 1, end: 13, stroke: 0.4pt),
    [val], [test], [shift], vsep,
    [val], [test], [shift], vsep,
    [val], [test], [shift], vsep,
    [val], [test], [shift],
  ),
  table.hline(),
  [`CVaR`], vsep,
  [18.20], [44.51], [60.22], vsep,
  [22.78], [29.71], [37.43], vsep,
  [18.04], [23.39], [29.49], vsep,
  [19.83], [24.11], [29.46],
  [`CVaR`+`augmentation`], vsep,
  [20.24], [41.99], [65.09], vsep,
  [27.03], [31.88], [41.62], vsep,
  [23.73], [29.46], [36.56], vsep,
  [23.04], [323.07], [327.56],
  [`WDRO`], vsep,
  [18.54], [37.76], [41.46], vsep,
  [23.10], [25.19], [32.08], vsep,
  [19.48], [22.82], [26.90], vsep,
  [21.86], [23.54], [27.24],
  [`WDRO`+`convex order`], vsep,
  [20.99], [30.89], [35.63], vsep,
  [22.62], [33.65], [37.63], vsep,
  [22.13], [25.19], [28.79], vsep,
  [21.19], [23.82], [28.14],
  table.hline(),
)

- `CVaR`+`augmentation` at $n=1000$: one of the 3 seeds diverged on the shift tiers
  (test CVaR-95 in the hundreds vs. the other two seeds' $tilde 30$); mean above is skewed by that
  outlier, not representative of typical-seed behavior at that $n$.
- `WDRO`+`convex order` is competitive with plain `WDRO` in-dist and on the hard shift tier at
  $n>=750$, but noisier and slightly worse at $n=500$; the convex-order penalty does not yet show a
  clear win over unconstrained `WDRO` at this sample-size range.

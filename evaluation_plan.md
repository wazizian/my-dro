# CVaR versus WDRO Benchmark and Distribution Evaluation Plan

## 1. Goal

Benchmark three hedging methods:

1. Nominal CVaR deep hedging.
2. Entropic WDRO deep hedging.
3. Joint-CVaR WDRO deep hedging.

The benchmark must determine:

- Whether robust training reduces risk under distribution shift.
- How much in-distribution performance it sacrifices.
- Whether losses become more concentrated, more dispersed, more skewed, or
  heavier-tailed.
- Whether conclusions persist across seeds, markets, transaction costs, and
  stress types.

The three native objectives have different scales. Never rank methods by their
training losses. Rank them using common pathwise terminal losses evaluated on
identical paths.

## 2. Models

### Nominal CVaR

Use the existing Rockafellar--Uryasev loss at alpha 0.95:

\[
q+\frac{1}{1-\alpha}\mathbb E[(L-q)_+].
\]

### Entropic WDRO

Use the implemented KL dual:

\[
\lambda\rho+
a\mathbb E_{\xi\sim P}
\left[
\log\mathbb E_{\xi'\sim\pi_0(\cdot\mid\xi)}
\exp\left(
\frac{f_\theta(\xi')-\lambda c(\xi,\xi')}{a}
\right)
\right],
\qquad
a=\epsilon+\lambda\tau.
\]

Use the implemented biased Monte Carlo log-mean-exp estimator.

### Joint-CVaR WDRO

Use p equal to 1, q equal to infinity, and one global scalar eta:

\[
\lambda\rho+\eta+
\frac{1}{1-\alpha}
\mathbb E_{\xi,\xi'}
\left[
(f_\theta(\xi')-\lambda c(\xi,\xi')-\eta)_+
\right].
\]

Fix alpha at 0.95 for the primary comparison.

## 3. Primary experiment matrix

Run all three models under:

| Market | Transaction cost |
|---|---:|
| Black--Scholes | 0 |
| Black--Scholes | 0.01 |
| Heston | 0 |
| Heston | 0.01 |

Use:

- The same network architecture.
- The same strategy initialization for a given seed.
- The same batch size.
- Common base-path keys where possible.
- Common validation and test paths.
- Final training seeds 0, 1, 2, 3, and 4.
- Heston float64 for final reported runs.

The primary comparison uses equal optimizer-update budgets. Also report an
equal-wall-clock comparison because WDRO updates process multiple proposals.

## 4. Data and random-key separation

Use separate deterministic key families for:

1. Network initialization.
2. Coordinate standardization.
3. Online training paths.
4. Training proposal noise.
5. Validation paths.
6. Validation proposal noise.
7. In-distribution test paths.
8. Each parametric stress.
9. Coordinate and adversarial stress.
10. Bootstrap resampling.

Do not use final test paths for tuning. Freeze one evaluation coordinate system
per market for common coordinate stresses. Do not evaluate each model using a
different stress coordinate system.

## 5. Fair loss views

For each strategy retain pathwise loss

\[
L=Z-G_T(\delta)+C_T(\delta).
\]

Evaluate three views.

### Unpriced loss

Set strategy cash to zero. This is the primary view because it measures hedge
quality without translating each model by a different price.

### Median-centered loss

\[
\widetilde L=L-\operatorname{median}(L).
\]

This isolates spread, concentration, skewness, and tail shape from location.

### Price-adjusted loss

\[
L^{priced}=L-p_{model}.
\]

This measures residual risk after charging the model's independently estimated
indifference price.

Every table and plot must identify which loss view it uses.

## 6. Evaluation distributions

### 6.1 In-distribution

Use independent paths from the training market:

- 100,000 paths during selection.
- 1,000,000 paths for final selected configurations.
- Identical chunks for all models.

### 6.2 Black--Scholes volatility shifts

Evaluate volatility:

    0.15, 0.20, 0.25, 0.30

The 0.20 case is in-distribution.

### 6.3 Black--Scholes rate shifts

Evaluate risk-free rate:

    0.00, 0.02, 0.05, 0.08

The 0.00 case is in-distribution and matches the current zero-rate simulator.

This requires adding a `rate` field to `MarketConfig` (default `0.0`) and
incorporating it into the Black--Scholes drift:

\[
d\log S_t=\left(r-\tfrac12\sigma^2\right)dt+\sigma\,dW_t.
\]

Apply the shift only to the market path simulator. Leave discounting and
strategy accounting as currently implemented unless a separate discounting
review changes them. Cross with the volatility grid in section 6.2 only for
the frozen moderate/severe combined scenarios, not as a full Cartesian
product, to keep the stress grid tractable.

### 6.4 Heston shifts

Use one-at-a-time shifts:

    long_variance:    0.02, 0.04, 0.06, 0.08
    vol_of_variance:  1.0,  2.0,  3.0
    correlation:     -0.9, -0.7, -0.4
    kappa:            0.5,  1.0,  2.0

Also freeze two combined scenarios before final testing:

    moderate:
      long_variance = 0.06
      vol_of_variance = 2.5
      correlation = -0.8
      kappa = 0.75

    severe:
      long_variance = 0.08
      vol_of_variance = 3.0
      correlation = -0.9
      kappa = 0.5

### 6.5 Transaction-cost misspecification

Evaluate each fixed strategy with realized costs:

    0, 0.005, 0.01, 0.02

This separates market-path robustness from execution-cost robustness.

### 6.6 Random coordinate stress

In shared standardized trajectory coordinates, add Gaussian perturbations with
scales:

    0.025, 0.05, 0.10, 0.20

Use common perturbation noise and decode to valid market trajectories.

### 6.7 Transport-radius stress

The current cost is

\[
c(\xi,\xi')
=\frac12
\left\|
\frac{\xi'-\xi}{\sigma_0}
\right\|_2^2.
\]

It grows with coordinate dimension D. Define normalized radius

\[
\bar\rho=\frac{2\rho}{D},
\qquad
\rho=\bar\rho D/2.
\]

Evaluate:

    rho_bar = 0, 0.003, 0.01, 0.03, 0.10, 0.30

Report both raw and normalized radius.

### 6.8 Common adversarial stress

Implement an evaluation-only projected-gradient attack:

1. Start from each test trajectory in standardized coordinates.
2. Maximize the fixed strategy's terminal loss.
3. Project onto the requested quadratic-cost ball.
4. Decode into a valid market path.
5. Use identical attack budgets and restarts for all models.

Validation grid:

    attack steps:       10, 25, 50
    step-size ratio:    0.10, 0.25, 0.50 of radius
    random restarts:    1, 4

Select attack settings by validation convergence, not by which model they harm.
Label this as an adversarial diagnostic rather than an exact WDRO solver.

## 7. ULA proposal calibration

Test only:

    k = 0, 1, 2

### 7.1 Proposal grid

    proposal_scale:
      0.025, 0.05, 0.10, 0.20

    step-size ratio h / proposal_scale^2:
      0.25, 0.50, 1.0, 2.0

    ula_gradient_clip:
      5, 10, 50

    proposal_samples:
      16, 32, 64

Always set

\[
h=r_h\sigma_0^2.
\]

Do not compare raw ULA step sizes while changing proposal scale.

### 7.2 Proposal diagnostics

Record:

- Mean and median ESS.
- ESS divided by proposal count.
- Maximum normalized importance weight.
- Log-weight mean, standard deviation, minimum, and maximum.
- Fraction of effectively zero weights.
- Mean transport cost.
- Mean coordinate displacement.
- Proposed-loss change relative to k equal to zero.
- Fraction of clipped ULA gradients.
- Nonfinite proposal, loss, weight, and gradient counts.
- Runtime per update.

Reject configurations when:

- Any nonfinite quantity occurs.
- Median normalized ESS is below 0.20.
- More than 5% of batches have normalized ESS below 0.10.
- One proposal carries over 95% of weight in more than 5% of batches.
- A positive k produces no measurable movement or loss change.

Prefer useful proposal movement with median normalized ESS between approximately
0.30 and 0.80.

### 7.3 Calibration order

Avoid the full Cartesian product:

1. Screen proposal scale at k equal to zero.
2. Keep two scales and screen k and step-size ratio.
3. Tune gradient clipping.
4. Tune proposal count last.

Use 2,000 steps and two seeds for this stage.

## 8. Robust radius and objective tuning

### 8.1 Radius

Tune normalized radius:

    0.003, 0.01, 0.03, 0.10, 0.30

Convert to raw radius separately for Black--Scholes and Heston.

### 8.2 Entropic WDRO

Primary grid:

    tau = 0
    kl_epsilon = 0.03, 0.10, 0.30

Only if this grid is unstable or dominated, run a separately reported secondary
grid:

    tau = 0.01, 0.10

### 8.3 Joint-CVaR WDRO

Use:

    alpha = 0.95
    rho_bar = 0.003, 0.01, 0.03, 0.10, 0.30

Initialize eta from a pilot nominal 95% loss quantile. During screening compare
this with zero initialization, then keep one rule consistently.

### 8.4 Optimizer tuning

Give every model the same optimizer search:

    learning_rate = 0.001, 0.0025, 0.005
    updates = 10,000, 20,000, 40,000

Track:

- Strategy gradient norm.
- Lambda and its gradient.
- Eta and its gradient for joint CVaR.
- Native validation objective.
- Common validation CVaR.
- Common stressed-validation CVaR.
- Proposal ESS.

More steps are justified only by stable common validation risk, not by a lower
native training objective.

## 9. Successive halving

| Stage | Steps | Seeds | Validation paths | Retention |
|---|---:|---:|---:|---:|
| Proposal screening | 2,000 | 2 | 16,384 | top 25% valid |
| Radius/objective screening | 5,000 | 2 | 32,768 | top 5 |
| Promotion | 10,000 | 3 | 65,536 | top 2 |
| Final training | 20,000--40,000 | 5 | 100,000 | one selected |
| Final testing | fixed | 5 | 1,000,000 | report all |

Count rejected and failed runs in the computational budget. Give nominal CVaR
the same learning-rate and duration tuning budget.

## 10. Model selection

Use common validation pathwise losses with cash zero. Select the configuration
with lowest worst-stress CVaR95 subject to

\[
\operatorname{CVaR}_{95}^{ID}(candidate)
\le
1.10
\operatorname{CVaR}_{95}^{ID}(best\ nominal).
\]

This permits at most a 10% in-distribution CVaR penalty.

If no robust candidate is admissible:

1. Do not relax the threshold silently.
2. Report the in-distribution versus stressed-risk Pareto frontier.
3. Mark the method as having no admissible selected configuration.

Never select from native entropic and joint-CVaR objectives directly.

## 11. Whole-distribution outputs

Compute every metric for all loss views, scenarios, and final seeds.

### 11.1 Full curves

Save:

- Empirical CDF.
- Survival function on a logarithmic vertical scale.
- Quantile function.
- Quantile difference from nominal CVaR:

  \[
  \Delta Q(p)=Q_{model}(p)-Q_{nominal}(p).
  \]

- Stop-loss curve:

  \[
  SL(t)=\mathbb E[(L-t)_+].
  \]

- CVaR spectrum:

  \[
  \alpha\mapsto\operatorname{CVaR}_\alpha(L).
  \]

- Median-centered concentration curve:

  \[
  r\mapsto P(|L-\operatorname{median}(L)|\le r).
  \]

Use reported quantiles:

    0.001, 0.005, 0.01, 0.025, 0.05, 0.10,
    0.25, 0.50, 0.75, 0.90, 0.95, 0.975,
    0.99, 0.995, 0.999

Use at least 199 probability points for plotted quantile curves.

### 11.2 Location

Report:

- Mean.
- Median.
- KDE mode, with bandwidth recorded.
- Probability of nonpositive loss.
- Probability of positive loss.
- Expected loss conditional on positive loss.
- Indifference price separately.

### 11.3 Spread

Report:

- Variance and standard deviation.
- Mean absolute deviation.
- Median absolute deviation.
- Interquartile range.
- Interdecile range.
- Central 90%, 95%, and 99% interval widths.
- Gini mean difference:

  \[
  \mathbb E|L-L'|.
  \]

Treat MAD and IQR as primary spread summaries because variance is
outlier-sensitive.

### 11.4 Peakedness and concentration

Report:

- Concentration probabilities around the median.
- Radius containing 25%, 50%, 75%, 90%, 95%, and 99% of losses.
- Density at the median with a fixed KDE bandwidth rule.
- Density at the estimated mode.
- Excess kurtosis only as a secondary diagnostic.

Do not infer peakedness from kurtosis alone. A distribution is more centrally
concentrated when its centered concentration curve is higher and its central
coverage radii are smaller.

### 11.5 Asymmetry

Report:

- Classical skewness.
- Bowley skewness:

  \[
  \frac{Q_{0.75}+Q_{0.25}-2Q_{0.50}}
       {Q_{0.75}-Q_{0.25}}.
  \]

- Upper spread \(Q_{0.95}-Q_{0.50}\).
- Lower spread \(Q_{0.50}-Q_{0.05}\).
- Upper-to-lower spread ratio.

### 11.6 Tails

Report:

- VaR at 90%, 95%, 97.5%, 99%, and 99.5%.
- CVaR at the same levels.
- Conditional variance above each VaR.
- Mean excess curve:

  \[
  e(t)=\mathbb E[L-t\mid L>t].
  \]

- Tail probabilities above fixed monetary thresholds.
- First and second lower partial moments:

  \[
  \mathbb E[(L-t)_+],
  \qquad
  \mathbb E[(L-t)_+^2].
  \]

- Maximum loss and mean of the worst 10, 100, and 1,000 paths, marked as
  sample-size-sensitive.

### 11.7 Distribution distances

Compare each robust method with nominal CVaR using:

- Wasserstein-1.
- Wasserstein-2.
- Kolmogorov--Smirnov.
- Energy distance.

Compute distances for raw and median-centered losses. Raw distance mixes
location and shape; centered distance isolates spread and shape.

### 11.8 Dominance diagnostics

Check:

- First-order empirical dominance across the quantile grid.
- Upper-tail dominance for probabilities at least 0.90.
- Stop-loss dominance across the threshold grid.

Claim dominance only when simultaneous bootstrap bands support the inequality
over the stated region.

## 12. Evolution over training, radius, and stress

Evaluate available checkpoints at:

    0, 500, 1,000, 2,000, 5,000, 10,000, 20,000, 40,000 updates

Produce:

- Quantile fan chart over training.
- Training-step by percentile heatmap.
- Radius by percentile heatmap.
- Stress-severity by percentile heatmap.
- CVaR spectra at selected checkpoints.
- Stop-loss curves at selected checkpoints.
- Centered ECDF and KDE overlays.
- QQ plots against nominal CVaR.
- Quantile-difference curves with confidence bands.
- Concentration curves with confidence bands.

Use fixed axes within each market/cost setting.

## 13. Interpretation rules

Use explicit conclusions:

- Less risky in the upper tail: upper quantiles, CVaR spectrum, and stop-loss
  values are lower.
- Less spread: centered MAD, IQR, and central interval widths are lower.
- More peaked: centered concentration curve is higher and coverage radii are
  lower.
- More right-skewed: Bowley skewness and upper/lower spread ratio are higher.
- Shifted lower: raw quantiles improve while centered shape stays similar.
- Tail compression with central widening: upper-tail risk improves while IQR
  or MAD grows.

Do not call one model simply better when central and tail metrics disagree.

## 14. Statistical inference

Use paired comparisons based on common paths.

### Path uncertainty

- At least 1,000 paired bootstrap replicates for final results.
- 95% intervals for scalar metric differences.
- Simultaneous bands for ECDF, quantile difference, CVaR spectrum,
  concentration, and stop-loss curves.

### Training uncertainty

Across five seeds report:

- Mean.
- Standard deviation.
- Standard error.
- Median.
- Minimum and maximum.
- Paired per-seed difference from nominal CVaR.

One million paths from one model do not replace multiple training seeds.

### Multiplicity

Predeclare the primary outcome:

    worst-stress CVaR95 of unpriced loss

Treat remaining measures as secondary or distributional diagnostics. Apply
Holm correction within a metric family if formal p-values are reported.

## 15. Primary and secondary scores

Primary:

\[
\max_{s\in\mathcal S}\operatorname{CVaR}_{0.95}(L_s).
\]

Secondary:

- In-distribution CVaR95.
- Mean stressed CVaR95.
- Worst-stress CVaR99.
- Relative risk degradation.
- Area under the normalized-radius CVaR curve.
- Area under the stop-loss curve.
- Area between centered concentration curves.
- Runtime and peak memory.

Use a fixed small denominator stabilizer for relative degradation.

## 16. Computational reporting

Record:

- Hardware and JAX backend.
- Dtype.
- Compilation time.
- Training and evaluation time.
- Peak memory.
- Optimizer updates.
- Base paths processed.
- Proposal paths processed.
- Rejected and failed candidates.
- Effective proposal paths after ESS adjustment.

Report equal-update and equal-wall-clock comparisons.

## 17. Implementation files

### benchmark.py

- Parse a Tyro benchmark configuration.
- Generate deterministic run manifests.
- Train or load checkpoints.
- Run fixed stress scenarios.
- Reset cash for common comparison.
- Save pathwise losses and metadata.
- Perform successive-halving promotion.

### benchmark_metrics.py

- Scalar distribution metrics.
- Quantile, CVaR, stop-loss, and concentration curves.
- Distribution distances.
- Paired bootstrap intervals and bands.
- Selection constraints and Pareto fronts.

### benchmark_plots.py

- ECDF and survival plots.
- Quantile and quantile-difference plots.
- CVaR spectrum.
- Stop-loss and concentration curves.
- QQ plots.
- Fan charts, heatmaps, and Pareto plots.

### test_benchmark.py

- Formula tests.
- Paired comparison tests.
- Deterministic selection tests.
- Tiny end-to-end benchmark.

Reuse simulation, strategy, accounting, and WDRO code. Do not duplicate it.

## 18. Artifact layout

    benchmark/
      manifest.json
      tuning/
        raw_runs.csv
        rejected_runs.csv
        promotions.json
        selected_configs.json
      final/
        scalar_metrics.csv
        paired_differences.csv
        seed_summary.csv
        distribution_curves.npz
        pathwise_losses/
          scenario/model/seed.npz
        plots/
          ecdf/
          survival/
          quantiles/
          cvar_spectrum/
          stop_loss/
          concentration/
          qq/
          heatmaps/
          pareto/
      report.md

Store large pathwise arrays in compressed NumPy files, not JSON.

Every result row must include model, market, training and evaluation costs,
scenario, seeds, steps, learning rate, raw and normalized radius, objective
parameters, proposal settings, checkpoint path, and Git commit.

## 19. Pytest plan

Test:

1. Common paths are identical across models.
2. Common evaluation resets cash to zero.
3. Centered and price-adjusted losses are correct.
4. Radius conversion is correct for both market dimensions.
5. VaR and CVaR match known samples.
6. Quantile curves match NumPy.
7. Stop-loss values match known samples.
8. Concentration curves are monotone.
9. Coverage radii are monotone.
10. MAD, IQR, Bowley skewness, and tail widths match known samples.
11. Wasserstein distances vanish for identical samples.
12. Symmetric distances are symmetric.
13. Paired bootstrap uses common resampled indices.
14. Nonfinite candidates are rejected.
15. Low-ESS proposals are rejected.
16. The 10% in-distribution constraint is enforced.
17. Pareto extraction handles dominated and tied candidates.
18. Successive-halving promotion is deterministic.
19. Checkpoint loading preserves strategy and robust state.
20. A tiny three-model benchmark writes all required artifacts.

Run:

    uv run pytest

## 20. Acceptance criteria

The benchmark is complete only when:

1. Architecture and initialization are fair.
2. Search budgets are recorded and comparable.
3. Proposal settings pass ESS and finiteness gates.
4. Radius is normalized for coordinate dimension.
5. Selection uses common validation risk.
6. Five seeds finish for every primary setting.
7. Final models use identical test and stress paths.
8. Final evaluation uses one million paths per selected model/scenario.
9. Whole-distribution curves and scalar metrics are saved.
10. Paired path and training-seed uncertainty are reported.
11. Equal-update and equal-wall-clock comparisons are included.
12. All pytest tests pass.
13. The report discusses central-versus-tail tradeoffs.

## 21. Execution order

1. Implement common checkpoint evaluation.
2. Implement scalar distribution metrics.
3. Implement ECDF, quantile, stop-loss, CVaR, and concentration curves.
4. Implement paired bootstrap intervals and simultaneous bands.
5. Implement shared parametric stresses.
6. Implement coordinate and projected-gradient stresses.
7. Implement proposal diagnostics and rejection.
8. Run short proposal calibration.
9. Run radius and objective successive halving.
10. Tune optimizer settings for all methods.
11. Freeze selected configurations and final scenario manifest.
12. Run five-seed final training.
13. Run common one-million-path final evaluation.
14. Generate tables, curves, confidence intervals, and Pareto plots.
15. Audit artifacts and run the complete pytest suite.

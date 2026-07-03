# Deep Hedging in JAX

This reproduces the Black--Scholes and Heston settings from Buehler et al.'s
*Deep Hedging* with their semi-recurrent network architecture and Adam settings.
The objective is CVaR at level α=0.95 (Rockafellar--Uryasev formulation). The
`cash` field reports the CVaR indifference price: the initial capital such that
CVaR of the terminal loss equals zero.

## Setup and tests

```bash
uv sync
uv run pytest
```

## Four experiments

```bash
uv run python deep_hedging.py --market.name black_scholes --epsilon 0
uv run python deep_hedging.py --market.name black_scholes --epsilon 0.01
JAX_ENABLE_X64=1 uv run python deep_hedging.py --market.name heston --market.dtype float64 --epsilon 0
JAX_ENABLE_X64=1 uv run python deep_hedging.py --market.name heston --market.dtype float64 --epsilon 0.01
```

Heston defaults to float32 for speed; the commands above demonstrate the
configurable float64 stability mode. A stock-only Heston sanity diagnostic is
available with `--strategy.stock-only` and should be compared with the two-asset
run using the same seed and training budget.

Paper-scale evaluation adds `--evaluation.paper-scale`. For a quick smoke run,
use `--training.steps 5 --training.validation-size 256 --evaluation.paths 256`.
The optional paper cost grid is `0.015625, 0.0078125, 0.00390625, 0.001953125,
0.0009765625`; run the desired market once per value with `--epsilon VALUE`.
The tail level can be changed with `--alpha VALUE` (default 0.95).

## WDRO training

Two regularized WDRO duals are available: `entropic` and `joint_cvar`. The
joint-CVaR formulation uses one global scalar threshold. It does not implement
the conditional formulation with one threshold per base trajectory.

```bash
uv run python deep_hedging.py --robust.enabled --robust.objective entropic
uv run python deep_hedging.py --robust.enabled --robust.objective joint_cvar
```

The conditional reference kernel is Gaussian in standardized trajectory
coordinates. Set `--robust.ula-steps 0`, `1`, or `2` to use the reference
proposal or an augmented-space ULA proposal. For `k > 0`, importance weights
use the explicit final Gaussian transition density; the intractable endpoint
marginal is not evaluated. The default ULA step variance equals the default
reference variance (`0.01 = 0.1²`); when changing the proposal scale, start by
setting `--robust.ula-step-size` to its square.

Robust runs save their dual state, including fixed coordinate statistics, in
`robust.eqx` and append results to `outputs/robust_comparison.csv`.

Each run writes its configuration, metrics, training history, Equinox checkpoint,
loss curve, hedging-error histogram, holdings plot, and turnover plot below
`outputs/`. The Black--Scholes no-cost case also writes an analytic-delta
diagnostic. `outputs/comparison.csv` is updated across runs made with the common
evaluation-seed convention.

A summary table across all completed runs can be printed with:

```bash
uv run python summarize.py
```

## Results

Seed-0, 20,000-step runs under the CVaR objective (α=0.95):

```
market            ε         MSE         CVaR        cash      mean_cost   turnover    Δ corr
----------------  --------  ----------  ----------  --------  ----------  ----------  --------
black_scholes     0         5.418       3.127       3.127     0           2.32        0.9924
black_scholes     0.01      13.58       4.839       4.839     1.344       1.343       -
heston            0         3.914       4.187       4.187     0           1077        -
heston            0.01      8.057       5.598       5.598     0.8656      647.3       -
```

Turnover falls when transaction costs are added in both markets (2.32 → 1.34 for
Black--Scholes, 1077 → 647 for Heston), reproducing the expected direction. The
CVaR indifference price for no-cost Black--Scholes is `3.127` versus the analytic
risk-neutral price `2.287`; the gap is the risk premium charged by CVaR over the
risk-neutral measure. The learned-delta correlation is `0.9924`.

Under the earlier MSE objective with jointly fitted cash, average transaction
costs were absorbed by the cash intercept so the cost-reduction incentive was
absent. Under the CVaR objective, cash is excluded from training and recovered
afterward as the indifference price, so the full cost distribution enters the
strategy gradient.

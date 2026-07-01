"""Print a summary table of all completed experiment runs."""

import json
from pathlib import Path

OUTPUTS = Path("outputs")

COLS = [
    ("market",    "market",    "<16"),
    ("ε",         "epsilon",   "<8"),
    ("MSE",       "mse",       "<10"),
    ("CVaR",      "cvar",      "<10"),
    ("cash",      "cash",      "<8"),
    ("mean_cost", "mean_cost", "<10"),
    ("turnover",  "turnover",  "<10"),
    ("Δ corr",    "delta_corr","<8"),
]


def load_run(directory: Path) -> dict | None:
    config_path = directory / "config.json"
    metrics_path = directory / "metrics.json"
    if not config_path.exists() or not metrics_path.exists():
        return None
    config = json.loads(config_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    return {
        "market":     config["market"]["name"],
        "epsilon":    config["epsilon"],
        "mse":        metrics.get("mse"),
        "cvar":       metrics.get("cvar"),
        "cash":       metrics.get("cash"),
        "mean_cost":  metrics.get("mean_cost"),
        "turnover":   metrics.get("mean_turnover"),
        "delta_corr": metrics.get("analytic_delta_correlation"),
        "_dir":       directory.name,
    }


def fmt(value, spec) -> str:
    if value is None:
        return format("-", spec)
    if isinstance(value, float):
        return format(f"{value:.4g}", spec)
    return format(str(value), spec)


runs = sorted(
    filter(None, (load_run(d) for d in OUTPUTS.iterdir() if d.is_dir())),
    key=lambda r: (r["market"], r["epsilon"]),
)

if not runs:
    print("No completed runs found in outputs/")
else:
    header = "  ".join(format(label, spec) for label, _, spec in COLS)
    sep = "  ".join("-" * int(spec.lstrip("<>")) for _, _, spec in COLS)
    print(header)
    print(sep)
    for run in runs:
        row = "  ".join(fmt(run[key], spec) for _, key, spec in COLS)
        print(row)

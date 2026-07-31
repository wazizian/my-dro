"""Summarize an lsv_sweep.py results CSV into a per-(method, n) mean+/-std table (see lsv.md)."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ("kfold_val_cvar95", "test_cvar95_indist", "test_cvar95_weak",
          "test_cvar95_medium", "test_cvar95_hard")


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read the sweep CSV; input a path, output a list of raw string-valued row dicts."""

    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, tuple[float, float, int]]]:
    """Group rows by (method, n) and compute mean/std/seed-count per metric; input rows, output summary dict."""

    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], int(row["n"]))].append(row)
    summary: dict[tuple[str, int], dict[str, tuple[float, float, int]]] = {}
    for key, group_rows in groups.items():
        metric_stats = {}
        for metric in METRICS:
            values = [float(row[metric]) for row in group_rows if row.get(metric)]
            if not values:
                continue
            mean = statistics.mean(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            metric_stats[metric] = (mean, std, len(values))
        summary[key] = metric_stats
    return summary


def format_table(summary: dict[tuple[str, int], dict[str, tuple[float, float, int]]]) -> str:
    """Render the summary as an aligned text table; input summary dict, output a printable string."""

    header = ["method", "n", "seeds", "val_cvar95", "test_indist", "test_weak", "test_medium", "test_hard"]
    rows = []
    for (method, n), metric_stats in sorted(summary.items(), key=lambda item: (item[0][0], item[0][1])):
        seeds = max((count for _, _, count in metric_stats.values()), default=0)
        cell = lambda metric: (f"{metric_stats[metric][0]:.2f}±{metric_stats[metric][1]:.2f}"
                               if metric in metric_stats else "-")
        rows.append([method, str(n), str(seeds), cell("kfold_val_cvar95"), cell("test_cvar95_indist"),
                    cell("test_cvar95_weak"), cell("test_cvar95_medium"), cell("test_cvar95_hard")])
    widths = [max(len(header[i]), *(len(row[i]) for row in rows)) if rows else len(header[i])
             for i in range(len(header))]
    lines = ["  ".join(header[i].ljust(widths[i]) for i in range(len(header)))]
    lines.append("  ".join("-" * widths[i] for i in range(len(header))))
    for row in rows:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(len(header))))
    return "\n".join(lines)


def main() -> None:
    """Build and print (and optionally save) the results table from the CLI; inputs argv, output stdout/file."""

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, default=None,
                        help="also write the rendered table to this path")
    args = parser.parse_args()
    rows = load_rows(args.csv_path)
    table = format_table(summarize(rows))
    print(table)
    if args.output is not None:
        args.output.write_text(table + "\n")


if __name__ == "__main__":
    main()

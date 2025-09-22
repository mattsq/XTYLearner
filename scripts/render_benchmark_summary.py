#!/usr/bin/env python3
"""Render benchmark results into clean Markdown tables.

This utility reads the aggregated benchmark output produced by the
``aggregate_results.py`` helper (``current-benchmarks.json``) and
converts it into a human friendly Markdown report.  The previous
summary mixed every metric into a single, very wide table which became
hard to read once columns with long names were added.  The formatter
below organises the data into one table per metric so that values stay
aligned and easy to compare.

Usage
-----

.. code-block:: bash

    python scripts/render_benchmark_summary.py \
        --input current-benchmarks.json \
        --output benchmark-summary.md

The resulting Markdown can be appended to ``$GITHUB_STEP_SUMMARY`` or
attached to a pull request comment from the GitHub Actions workflow.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class MetricInfo:
    """Metadata describing how to display a benchmark metric."""

    key: str
    label: str
    better: str  # ``"lower"`` or ``"higher"``
    value_format: str

    @property
    def sort_reverse(self) -> bool:
        """Return whether the metric should be sorted in descending order."""

        return self.better == "higher"


METRICS: Tuple[MetricInfo, ...] = (
    MetricInfo(
        key="val_outcome_rmse",
        label="Validation Outcome RMSE (lower is better)",
        better="lower",
        value_format="{:.4f}",
    ),
    MetricInfo(
        key="val_treatment_accuracy",
        label="Validation Treatment Accuracy (higher is better)",
        better="higher",
        value_format="{:.3f}",
    ),
    MetricInfo(
        key="train_time_seconds",
        label="Training Time (seconds, lower is better)",
        better="lower",
        value_format="{:.2f}",
    ),
)


def _format_value(value: float, fmt: str) -> str:
    """Format numbers while guarding against NaNs and infinities."""

    if value is None or not math.isfinite(value):
        return "n/a"
    return fmt.format(value)


def _format_interval(interval: Iterable[float], fmt: str) -> str:
    """Format the confidence interval into ``a – b`` or ``n/a``."""

    try:
        low, high = interval
    except Exception:  # pragma: no cover - defensive for malformed JSON
        return "n/a"

    if not (math.isfinite(low) and math.isfinite(high)):
        return "n/a"
    return f"{fmt.format(low)} – {fmt.format(high)}"


def _normalise_result(result: Dict) -> Dict:
    """Extract a consistent record from a raw benchmark result item."""

    environment = result.get("environment", {})
    model = environment.get("model_name") or "unknown"
    dataset = environment.get("dataset_name") or "unknown"

    metric_key = None
    for info in METRICS:
        if result.get("name", "").endswith(info.key):
            metric_key = info.key
            break

    if metric_key is None:
        raise ValueError(f"Could not determine metric for result: {result.get('name')}")

    return {
        "metric": metric_key,
        "model": model,
        "dataset": dataset,
        "value": result.get("value"),
        "interval": result.get("range", []),
        "unit": result.get("unit", ""),
        "samples": result.get("samples"),
    }


def _sort_rows(rows: List[Dict], metric: MetricInfo) -> List[Dict]:
    """Sort rows by dataset name and metric value."""

    def sort_key(row: Dict) -> Tuple[str, float]:
        dataset = row.get("dataset") or ""
        value = row.get("value")
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            # ``NaN``/``None`` should always sink to the bottom regardless of order.
            return (dataset, math.inf)
        if metric.sort_reverse:
            return (dataset, -float(value))
        return (dataset, float(value))

    return sorted(rows, key=sort_key)


def build_tables(results: List[Dict]) -> str:
    """Create Markdown tables grouped by metric."""

    normalised = [_normalise_result(result) for result in results]

    sections: List[str] = []
    for info in METRICS:
        metric_rows = [row for row in normalised if row["metric"] == info.key]
        if not metric_rows:
            continue

        metric_rows = _sort_rows(metric_rows, info)

        header = [
            f"### {info.label}",
            "",
            "| Dataset | Model | Value | 95% CI | Unit | Samples |",
            "|:--|:--|--:|--:|:--|--:|",
        ]

        body_lines = []
        for row in metric_rows:
            value = _format_value(row["value"], info.value_format)
            interval = _format_interval(row.get("interval", []), info.value_format)
            unit = row.get("unit") or ""
            samples = row.get("samples")
            samples_str = str(samples) if isinstance(samples, int) else "n/a"
            body_lines.append(
                "| {dataset} | {model} | {value} | {interval} | {unit} | {samples} |".format(
                    dataset=row.get("dataset", "unknown"),
                    model=row.get("model", "unknown"),
                    value=value,
                    interval=interval,
                    unit=unit,
                    samples=samples_str,
                )
            )

        sections.extend(header + body_lines + [""])

    if not sections:
        return "_No benchmark results available._\n"

    return "\n".join(sections).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render benchmark results to Markdown tables")
    parser.add_argument("--input", required=True, help="Path to current-benchmarks.json")
    parser.add_argument("--output", required=True, help="Destination Markdown file")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")

    with open(input_path) as fh:
        data = json.load(fh)

    results = data.get("results", [])
    markdown = build_tables(results)

    output_path.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()

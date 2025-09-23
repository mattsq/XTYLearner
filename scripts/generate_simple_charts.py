#!/usr/bin/env python3
"""Generate benchmark visualisations and summary content for GitHub Pages."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class MetricInfo:
    """Metadata describing how to present a benchmark metric."""

    key: str
    label: str
    better: str  # "lower" or "higher"
    value_format: str

    @property
    def sort_reverse(self) -> bool:
        return self.better == "higher"


METRICS: Sequence[MetricInfo] = (
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

MAX_TREND_POINTS = 12


def _format_value(value: Optional[float], fmt: str) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return fmt.format(number)


def _format_interval(interval: Iterable[float], fmt: str) -> str:
    try:
        low, high = interval
    except Exception:
        return "n/a"

    try:
        low_value = float(low)
        high_value = float(high)
    except (TypeError, ValueError):
        return "n/a"

    if not (math.isfinite(low_value) and math.isfinite(high_value)):
        return "n/a"
    return f"{fmt.format(low_value)} – {fmt.format(high_value)}"


def _normalise_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    environment = result.get("environment") or {}
    model = environment.get("model_name")
    dataset = environment.get("dataset_name")

    name = result.get("name", "")
    name_parts = name.split("_", 2)
    if not model and name_parts:
        model = name_parts[0] or "unknown"
    if not dataset and len(name_parts) > 1:
        dataset = name_parts[1] or "unknown"

    metric_key: Optional[str] = None
    for info in METRICS:
        if name.endswith(info.key):
            metric_key = info.key
            break

    if metric_key is None:
        raise ValueError(f"Could not determine metric for result: {name}")

    return {
        "metric": metric_key,
        "model": model or "unknown",
        "dataset": dataset or "unknown",
        "value": result.get("value"),
        "interval": result.get("range", []),
        "unit": result.get("unit", ""),
        "samples": result.get("samples"),
    }


def _sort_rows(rows: List[Dict[str, Any]], metric: MetricInfo) -> List[Dict[str, Any]]:
    def sort_key(row: Dict[str, Any]) -> tuple[str, float]:
        dataset = row.get("dataset") or ""
        value = row.get("value")
        if value is None:
            numeric = math.inf
        else:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = math.inf
        if not math.isfinite(numeric):
            numeric = math.inf
        if metric.sort_reverse:
            numeric = -numeric
        return (dataset, numeric)

    return sorted(rows, key=sort_key)


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        if value.endswith("Z"):
            try:
                return datetime.fromisoformat(value[:-1])
            except ValueError:
                return None
        return None


TimeSeries = Dict[Tuple[str, str, str], List[Tuple[float, str, float]]]


def _build_time_series(history: Sequence[Mapping[str, Any]]) -> TimeSeries:
    series: TimeSeries = defaultdict(list)

    for idx, entry in enumerate(history):
        commit_full = entry.get("commit") or "unknown"
        commit = commit_full[:7]
        timestamp = _parse_timestamp(entry.get("timestamp"))
        order = timestamp.timestamp() if timestamp else float(idx)
        raw_results = entry.get("results", [])

        for result in raw_results:
            try:
                normalised = _normalise_result(result)
            except ValueError:
                continue

            value = normalised.get("value")
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric):
                continue

            key = (
                normalised["metric"],
                normalised["model"],
                normalised["dataset"],
            )
            series[key].append((order, commit, numeric))

    for key, points in series.items():
        points.sort(key=lambda item: item[0])
        series[key] = points[-MAX_TREND_POINTS:]

    return series


def _select_top_series(
    series: TimeSeries, metric: MetricInfo
) -> List[Tuple[Tuple[str, str], List[Tuple[str, float]]]]:
    candidates: List[Tuple[Tuple[str, str], float, List[Tuple[str, float]]]] = []

    for (metric_key, model, dataset), points in series.items():
        if metric_key != metric.key or not points:
            continue

        deduped: List[Tuple[str, float]] = []
        for _, commit, value in points:
            if deduped and deduped[-1][0] == commit:
                deduped[-1] = (commit, value)
            else:
                deduped.append((commit, value))

        if not deduped:
            continue

        latest_value = deduped[-1][1]
        candidates.append(((model, dataset), latest_value, deduped))

    reverse = metric.sort_reverse
    if reverse:
        candidates.sort(
            key=lambda item: (
                -item[1],
                str(item[0][0]).lower(),
                str(item[0][1]).lower(),
            )
        )
    else:
        candidates.sort(
            key=lambda item: (
                item[1],
                str(item[0][0]).lower(),
                str(item[0][1]).lower(),
            )
        )

    return [((model, dataset), points) for (model, dataset), _, points in candidates]


def _plot_trend_chart(
    metric: MetricInfo,
    series: List[Tuple[Tuple[str, str], List[Tuple[str, float]]]],
    output_dir: Path,
) -> Optional[str]:
    if not series:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    for (model, dataset), points in series:
        commits = [commit for commit, _ in points]
        values = [value for _, value in points]
        label = f"{model} on {dataset}"
        ax.plot(commits, values, marker="o", label=label)

    ax.set_title(f"{metric.label} over recent commits")
    ax.set_xlabel("Commit")
    ax.set_ylabel(metric.label)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    filename = f"benchmark_trend_{metric.key}.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return filename


def _build_trend_markdown(
    metric: MetricInfo,
    series: List[Tuple[Tuple[str, str], List[Tuple[str, float]]]],
) -> List[str]:
    if not series:
        return []

    lines = [f"#### {metric.label} trends", "", "| Model | Dataset | Latest Commit | Value | Δ vs previous |", "|:--|:--|:--|--:|--:|"]

    for (model, dataset), points in series:
        latest_commit, latest_value = points[-1]
        prev_value = points[-2][1] if len(points) > 1 else None
        delta = None if prev_value is None else latest_value - prev_value
        delta_str = "n/a" if delta is None else metric.value_format.format(delta)
        lines.append(
            "| {model} | {dataset} | {commit} | {value} | {delta} |".format(
                model=model,
                dataset=dataset,
                commit=latest_commit,
                value=metric.value_format.format(latest_value),
                delta=delta_str,
            )
        )

    lines.append("")
    return lines


def build_markdown_tables(results: Sequence[Dict[str, Any]]) -> str:
    sections: List[str] = []
    for info in METRICS:
        metric_rows = [row for row in results if row["metric"] == info.key]
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
            value = _format_value(row.get("value"), info.value_format)
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


def build_html_tables(results: Sequence[Dict[str, Any]]) -> str:
    sections: List[str] = []
    for info in METRICS:
        metric_rows = [row for row in results if row["metric"] == info.key]
        if not metric_rows:
            continue

        metric_rows = _sort_rows(metric_rows, info)
        rows_html = []
        for row in metric_rows:
            value = _format_value(row.get("value"), info.value_format)
            interval = _format_interval(row.get("interval", []), info.value_format)
            unit = row.get("unit") or ""
            samples = row.get("samples")
            samples_str = str(samples) if isinstance(samples, int) else "n/a"
            rows_html.append(
                "            <tr>\n"
                f"                <td>{escape(row.get('dataset', 'unknown'))}</td>\n"
                f"                <td>{escape(row.get('model', 'unknown'))}</td>\n"
                f"                <td>{escape(value)}</td>\n"
                f"                <td>{escape(interval)}</td>\n"
                f"                <td>{escape(unit)}</td>\n"
                f"                <td>{escape(samples_str)}</td>\n"
                "            </tr>"
            )

        sections.append(
            "        <section class=\"metric-section\">\n"
            f"            <h2>{escape(info.label)}</h2>\n"
            "            <table>\n"
            "                <thead>\n"
            "                    <tr>\n"
            "                        <th>Dataset</th>\n"
            "                        <th>Model</th>\n"
            "                        <th>Value</th>\n"
            "                        <th>95% CI</th>\n"
            "                        <th>Unit</th>\n"
            "                        <th>Samples</th>\n"
            "                    </tr>\n"
            "                </thead>\n"
            "                <tbody>\n"
            + "\n".join(rows_html)
            + "\n                </tbody>\n"
            "            </table>\n"
            "        </section>"
        )

    if not sections:
        return "        <p><em>No benchmark results available.</em></p>"

    return "\n".join(sections)


def _aggregate_metric(results: Sequence[Dict[str, Any]], metric_key: str) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = {}
    for row in results:
        if row["metric"] != metric_key:
            continue
        value = row.get("value")
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        model = row.get("model", "unknown")
        grouped.setdefault(model, []).append(numeric)

    aggregated: Dict[str, float] = {}
    for model, values in grouped.items():
        if values:
            aggregated[model] = sum(values) / len(values)
    return aggregated


def _plot_metric(
    axis: plt.Axes,
    series: Mapping[str, float],
    title: str,
    ylabel: str,
    *,
    reverse: bool = False,
) -> None:
    axis.set_title(title)
    if not series:
        axis.axis("off")
        axis.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        return

    if reverse:
        sorted_items = sorted(series.items(), key=lambda item: (-item[1], item[0]))
    else:
        sorted_items = sorted(series.items(), key=lambda item: (item[1], item[0]))
    models = [name for name, _ in sorted_items]
    values = [value for _, value in sorted_items]
    axis.bar(models, values)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", labelrotation=45)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")


def _render_html(
    latest: Optional[Mapping[str, Any]],
    summary: Dict[str, Any],
    html_tables: str,
    trend_sections: str,
) -> str:
    timestamp = latest.get("timestamp") if latest else None
    commit = latest.get("commit") if latest else None
    timestamp_str = timestamp or "n/a"
    commit_str = (commit or "unknown")[:8]

    models_list = summary.get("models", [])
    datasets_list = summary.get("datasets", [])

    models_list_html = "\n".join(
        f"                <li>{escape(item)}</li>" for item in models_list
    ) or "                <li><em>No models recorded.</em></li>"
    datasets_list_html = "\n".join(
        f"                <li>{escape(item)}</li>" for item in datasets_list
    ) or "                <li><em>No datasets recorded.</em></li>"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset=\"utf-8\">
        <title>XTYLearner Benchmark Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; color: #2c3e50; }}
            .header {{ border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }}
            .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 30px; }}
            .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; }}
            .summary-card {{ background: #ffffff; border-radius: 5px; padding: 16px; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08); }}
            .summary-card h3 {{ margin-top: 0; margin-bottom: 12px; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.05em; color: #6c7a89; }}
            .summary-card .value {{ font-size: 2rem; font-weight: bold; color: #2c3e50; }}
            .summary-card ul {{ margin: 12px 0 0; padding-left: 20px; }}
            .summary-card li {{ margin-bottom: 4px; }}
            .chart {{ text-align: center; margin: 30px 0; }}
            .chart img {{ max-width: 100%; height: auto; border: 1px solid #e1e1e1; border-radius: 6px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            section.metric-section {{ margin-bottom: 40px; }}
            footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; }}
        </style>
    </head>
    <body>
        <div class=\"header\">
            <h1>🚀 XTYLearner Model Benchmarks</h1>
            <p>Last updated: {escape(timestamp_str)} | Commit: {escape(commit_str)}</p>
        </div>

        <div class=\"summary\">
            <h2>📊 Summary</h2>
            <div class=\"summary-grid\">
                <div class=\"summary-card\">
                    <h3>Total Benchmarks</h3>
                    <div class=\"value\">{summary.get("total", 0)}</div>
                </div>
                <div class=\"summary-card\">
                    <h3>Models Tested</h3>
                    <div class=\"value\">{summary.get("models_count", 0)}</div>
                    <ul>
{models_list_html}
                    </ul>
                </div>
                <div class=\"summary-card\">
                    <h3>Datasets Used</h3>
                    <div class=\"value\">{summary.get("datasets_count", 0)}</div>
                    <ul>
{datasets_list_html}
                    </ul>
                </div>
            </div>
        </div>

        <div class=\"chart\">
            <h2>📈 Performance Overview</h2>
            <img src=\"benchmark_summary.png\" alt=\"Benchmark Summary\">
        </div>

{trend_sections}

{html_tables}

        <footer>
            <p>Generated by the XTYLearner Benchmark Suite.</p>
        </footer>
    </body>
    </html>
    """


def generate_charts(history_file: str, output_dir: str, markdown_output: Optional[str] = None) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    with open(history_file) as f:
        history = json.load(f)

    markdown_summary = "_No benchmark results available._\n"
    latest: Optional[Mapping[str, Any]] = None
    normalised: List[Dict[str, Any]] = []
    trend_images: Dict[str, str] = {}
    trend_markdown: List[str] = []

    if history:
        latest = history[-1]
        raw_results = latest.get("results", [])
        if raw_results:
            try:
                normalised = [_normalise_result(result) for result in raw_results]
            except ValueError as exc:
                print(f"Error normalising results: {exc}")
                normalised = []

    time_series = _build_time_series(history) if history else {}

    if time_series:
        for info in METRICS:
            selected_series = _select_top_series(time_series, info)
            if not selected_series:
                continue
            image_name = _plot_trend_chart(info, selected_series, output_path)
            if image_name:
                trend_images[info.key] = image_name
            trend_markdown.extend(_build_trend_markdown(info, selected_series))

    if normalised:
        markdown_summary = build_markdown_tables(normalised)
        html_tables = build_html_tables(normalised)
    else:
        html_tables = "        <p><em>No benchmark results available.</em></p>"

    rmse_series = _aggregate_metric(normalised, "val_outcome_rmse") if normalised else {}
    accuracy_series = _aggregate_metric(normalised, "val_treatment_accuracy") if normalised else {}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    _plot_metric(axes[0], rmse_series, "Model Outcome RMSE (↓)", "RMSE")
    _plot_metric(
        axes[1],
        accuracy_series,
        "Model Treatment Accuracy (↑)",
        "Accuracy",
        reverse=True,
    )
    fig.tight_layout()
    plt.savefig(output_path / "benchmark_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    models = []
    datasets = []
    if normalised:
        models = sorted({row.get("model", "unknown") for row in normalised})
        datasets = sorted({row.get("dataset", "unknown") for row in normalised})

    summary = {
        "total": len(normalised),
        "models_count": len(models),
        "datasets_count": len(datasets),
        "models": [item for item in models if item and item != "unknown"],
        "datasets": [item for item in datasets if item and item != "unknown"],
    }

    if latest:
        metadata = latest.get("metadata") or {}
        models_meta = metadata.get("models")
        datasets_meta = metadata.get("datasets")
        if isinstance(models_meta, list) and models_meta:
            summary["models"] = sorted(set(models_meta))
            summary["models_count"] = len(summary["models"])
        if isinstance(datasets_meta, list) and datasets_meta:
            summary["datasets"] = sorted(set(datasets_meta))
            summary["datasets_count"] = len(summary["datasets"])

    if trend_markdown:
        markdown_summary = markdown_summary.rstrip() + "\n\n" + "\n".join(trend_markdown)

    trend_sections_html = "        <p><em>No historical trend data available yet.</em></p>"
    trend_sections: List[str] = []
    for info in METRICS:
        image_name = trend_images.get(info.key)
        if not image_name:
            continue
        trend_sections.append(
            "        <div class=\"chart\">\n"
            f"            <h2>Trend – {escape(info.label)}</h2>\n"
            f"            <img src=\"{escape(image_name)}\" alt=\"Trend for {escape(info.label)}\">\n"
            "        </div>"
        )

    if trend_sections:
        trend_sections_html = "\n".join(trend_sections)

    html_content = _render_html(latest, summary, html_tables, trend_sections_html)
    (output_path / "index.html").write_text(html_content, encoding="utf-8")

    if markdown_output:
        Path(markdown_output).write_text(markdown_summary, encoding="utf-8")

    print(f"Charts and summary written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True, help="Path to benchmark history JSON file")
    parser.add_argument("--output-dir", required=True, help="Directory to write charts and HTML")
    parser.add_argument(
        "--markdown-output",
        help="Optional path to write a Markdown summary for the workflow run",
    )
    args = parser.parse_args()

    generate_charts(args.history, args.output_dir, args.markdown_output)

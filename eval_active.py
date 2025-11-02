#!/usr/bin/env python3
"""Benchmark active learning strategies across datasets and models."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from xtylearner.eval_active_runner import (
    run_active_benchmark_once,
    aggregate_across_seeds,
)


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:.4f}"
    if isinstance(value, (int, str)):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=False)
        fp.write("\n")


def _write_markdown(path: Path, results: List[Dict[str, Any]]) -> None:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for entry in results:
        grouped[(entry["dataset"], entry["model"])].append(entry)

    lines: List[str] = ["# Active Learning Benchmark Results", ""]
    for (dataset, model), entries in sorted(grouped.items()):
        header_entry = entries[0]
        budget = header_entry.get("budget")
        batch = header_entry.get("batch")
        epochs = header_entry.get("epochs_per_round")
        seeds = header_entry.get("seeds", [])
        lines.append(f"### {dataset} / {model}")
        lines.append(
            f"budget={budget}, batch={batch}, epochs_per_round={epochs}, seeds={seeds}"
        )
        lines.append("")

        aulc_metrics = sorted({m for e in entries for m in e.get("AULC", {}).keys()})
        final_metrics = sorted(
            {m for e in entries for m in e.get("final_metrics_mean", {}).keys()}
        )

        columns = ["strategy"]
        columns += [f"AULC_{m}" for m in aulc_metrics]
        columns += [f"Final_{m}" for m in final_metrics]

        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

        for entry in sorted(entries, key=lambda e: e["strategy"]):
            row = [entry["strategy"]]
            for metric in aulc_metrics:
                row.append(_format_value(entry.get("AULC", {}).get(metric)))
            for metric in final_metrics:
                row.append(_format_value(entry.get("final_metrics_mean", {}).get(metric)))
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("benchmark_config.json"),
        help="Path to benchmark configuration JSON",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("benchmark_active_results.json"),
        help="Where to store the aggregated JSON results",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("benchmark_active_results.md"),
        help="Where to store the Markdown summary",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    if "active_benchmark" not in config:
        raise KeyError("Configuration missing 'active_benchmark' section")
    active_cfg = config["active_benchmark"]

    datasets = active_cfg.get("datasets", [])
    models = active_cfg.get("models", [])
    strategies = active_cfg.get("strategies", [])
    seeds = active_cfg.get("seeds", [])
    budget = active_cfg.get("budget", 0)
    batch = active_cfg.get("batch", 1)
    epochs_per_round = active_cfg.get("epochs_per_round", 1)

    all_results: List[Dict[str, Any]] = []

    for dataset_conf in datasets:
        for model_conf in models:
            for strategy_conf in strategies:
                runs = []
                for seed in seeds:
                    run = run_active_benchmark_once(
                        dataset_conf,
                        model_conf,
                        strategy_conf,
                        budget,
                        batch,
                        epochs_per_round,
                        seed,
                    )
                    runs.append(run)

                agg = aggregate_across_seeds(runs)
                entry = {
                    "dataset": dataset_conf["name"],
                    "model": model_conf["name"],
                    "strategy": strategy_conf["name"],
                    "budget": budget,
                    "batch": batch,
                    "epochs_per_round": epochs_per_round,
                    "seeds": seeds,
                    **agg,
                    "runs": runs,
                }
                all_results.append(entry)

    _write_json(args.out_json, all_results)
    _write_markdown(args.out_md, all_results)


if __name__ == "__main__":
    main()


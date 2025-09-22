#!/usr/bin/env python3
"""Summarise benchmark timing information for job reporting."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_timings(current_path: Path, timing_dir: Path | None) -> List[Dict[str, Any]]:
    timings: List[Dict[str, Any]] = []

    if current_path.exists():
        with current_path.open() as fh:
            try:
                payload = json.load(fh)
            except json.JSONDecodeError:
                payload = {}
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        if isinstance(metadata, dict):
            source_timings = metadata.get("timings", [])
            if isinstance(source_timings, list):
                timings.extend(t for t in source_timings if isinstance(t, dict))

    if timing_dir and timing_dir.exists():
        for timing_file in sorted(timing_dir.rglob("timing-*.json")):
            with timing_file.open() as fh:
                try:
                    payload = json.load(fh)
                except json.JSONDecodeError:
                    continue
            if isinstance(payload, dict):
                timings.append(payload)

    return timings


def summarise_stages(timings: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    stage_names = [
        "environment_setup_seconds",
        "execution_step_seconds",
        "data_prep_seconds",
        "warmup_seconds",
        "benchmark_seconds",
        "train_time_seconds",
        "total_seconds",
    ]
    summary = {name: 0.0 for name in stage_names}
    count = 0
    for record in timings:
        count += 1
        for name in stage_names:
            value = record.get(name)
            if isinstance(value, (float, int)):
                summary[name] += float(value)
    if count:
        for name in stage_names:
            summary[name] /= count
    summary["count"] = float(count)
    return summary


def render_table(timings: List[Dict[str, Any]]) -> str:
    header = (
        "| Model | Dataset | Total (s) | Env Setup | Step Runtime | Data Prep | Warmup | Benchmark | Train Time |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rows = []
    for record in sorted(timings, key=lambda r: r.get("total_seconds", 0), reverse=True):
        row = (
            "| {model} | {dataset} | {total:.1f} | {env:.1f} | {step:.1f} | {prep:.1f} | {warmup:.1f} | {bench:.1f} | {train:.1f} |"
        ).format(
            model=record.get("model", "?"),
            dataset=record.get("dataset", "?"),
            total=float(record.get("total_seconds", 0.0) or 0.0),
            env=float(record.get("environment_setup_seconds", 0.0) or 0.0),
            step=float(record.get("execution_step_seconds", 0.0) or 0.0),
            prep=float(record.get("data_prep_seconds", 0.0) or 0.0),
            warmup=float(record.get("warmup_seconds", 0.0) or 0.0),
            bench=float(record.get("benchmark_seconds", 0.0) or 0.0),
            train=float(record.get("train_time_seconds", 0.0) or 0.0),
        )
        rows.append(row)
    return "\n".join([header, *rows]) if rows else "_No detailed timing records._"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--timing-dir", type=Path)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    timings = load_timings(args.current, args.timing_dir)

    if not timings:
        args.summary_out.write_text("_No timing data captured._\n")
        return

    stage_summary = summarise_stages(timings)
    lines = [
        "### Benchmark Timing Overview",
        f"Benchmarks analysed: {int(stage_summary.get('count', 0))}",
        f"Avg env setup: {stage_summary['environment_setup_seconds']:.2f}s",
        f"Avg benchmark step runtime: {stage_summary['execution_step_seconds']:.2f}s",
        f"Avg data prep: {stage_summary['data_prep_seconds']:.2f}s",
        f"Avg warmup: {stage_summary['warmup_seconds']:.2f}s",
        f"Avg benchmark loop: {stage_summary['benchmark_seconds']:.2f}s",
        f"Avg train time (sum of iterations): {stage_summary['train_time_seconds']:.2f}s",
        f"Avg total per benchmark: {stage_summary['total_seconds']:.2f}s",
        "",
        render_table(timings),
    ]
    args.summary_out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

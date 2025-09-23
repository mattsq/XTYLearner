#!/usr/bin/env python3
"""Update timing metadata artifacts for benchmark runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_timings(path: Path) -> list[object]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        return []
    timings = metadata.get("timings", [])
    return timings if isinstance(timings, list) else []


def select_record(timings: list[object], model: str, dataset: str) -> dict[str, object]:
    for entry in timings:
        if not isinstance(entry, dict):
            continue
        if entry.get("model") == model and entry.get("dataset") == dataset:
            return entry.copy()
    for entry in timings:
        if isinstance(entry, dict):
            return entry.copy()
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--env-seconds", type=float, default=0.0)
    parser.add_argument("--benchmark-seconds", type=float, default=0.0)
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    timings = read_timings(Path(args.results_path))
    record = select_record(timings, args.model, args.dataset)
    record.setdefault("model", args.model)
    record.setdefault("dataset", args.dataset)
    record["environment_setup_seconds"] = args.env_seconds
    record["execution_step_seconds"] = args.benchmark_seconds

    Path(args.output_path).write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

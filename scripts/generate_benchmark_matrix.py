#!/usr/bin/env python3
"""Helper to compute the benchmark matrix for GitHub Actions."""

from __future__ import annotations

import argparse
import json
import os
from typing import Sequence


PR_MATRIX = {
    "model": ["cycle_dual", "mean_teacher"],
    "dataset": ["synthetic"],
}

FULL_MATRIX = {
    "model": [
        "cycle_dual",
        "mean_teacher",
        "prob_circuit",
        "ganite",
        "flow_ssc",
        "vat",
        "fixmatch",
        "dragon_net",
        "diffusion_cevae",
        "bridge_diff",
        "cacore",
        "ss_cevae",
        "tab_jepa",
        "masked_tabular_transformer",
        "gnn_scm",
        "joint_ebm",
        "factor_vae_plus",
        "semiite",
        "vacim",
        "multitask",
    ],
    "dataset": ["synthetic", "synthetic_mixed"],
}

DEFAULT_MATRIX = {
    "model": [
        "cycle_dual",
        "mean_teacher",
        "prob_circuit",
        "ganite",
        "flow_ssc",
        "vat",
        "fixmatch",
        "dragon_net",
        "diffusion_cevae",
        "bridge_diff",
        "cacore",
        "ss_cevae",
    ],
    "dataset": ["synthetic", "synthetic_mixed"],
}


def as_bool(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def choose_matrix(event_name: str, full_benchmark: bool) -> dict[str, Sequence[str]]:
    if event_name == "pull_request":
        return PR_MATRIX
    if full_benchmark:
        return FULL_MATRIX
    return DEFAULT_MATRIX


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-name", default=os.environ.get("GITHUB_EVENT_NAME", ""))
    parser.add_argument(
        "--full-benchmark",
        default=os.environ.get("FULL_BENCHMARK", "false"),
        help="Whether to emit the full benchmark matrix",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("GITHUB_OUTPUT"),
        help="Path to the GitHub Actions output file",
    )
    args = parser.parse_args()

    matrix = choose_matrix(args.event_name, as_bool(args.full_benchmark))
    payload = json.dumps(matrix, separators=(",", ":"))

    if args.output:
        with open(args.output, "a", encoding="utf-8") as fh:
            fh.write(f"matrix={payload}\n")
    else:
        print(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Helper to compute the benchmark matrix for GitHub Actions."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Sequence, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


KNOWN_MODELS = set(FULL_MATRIX["model"]) | set(DEFAULT_MATRIX["model"]) | set(PR_MATRIX["model"])


def discover_module_model_map() -> dict[str, list[str]]:
    """Parse model files to determine which registry names they provide."""

    models_dir = REPO_ROOT / "xtylearner" / "models"
    pattern = re.compile(r"@register_model\(\s*['\"]([^'\"]+)['\"]\s*\)")
    module_map: dict[str, list[str]] = {}

    for path in models_dir.glob("**/*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        names = pattern.findall(text)
        if not names:
            continue

        module = (
            path.relative_to(REPO_ROOT)
            .with_suffix("")
            .as_posix()
            .replace("/", ".")
        )
        module_map[module] = names

    return module_map


MODULE_MODEL_MAP = discover_module_model_map()


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def files_to_models(files: Sequence[str]) -> List[str]:
    if not files:
        return []

    resolved: list[str] = []
    for file_path in files:
        if not file_path or not file_path.endswith(".py"):
            continue
        module = file_path.replace("/", ".").rsplit(".py", 1)[0]
        if module.endswith(".__init__"):
            module = module.rsplit(".", 1)[0]
        matches = MODULE_MODEL_MAP.get(module)
        if matches:
            resolved.extend(matches)

    return sorted({name for name in resolved if name in KNOWN_MODELS})


def choose_matrix(
    event_name: str,
    full_benchmark: bool,
    changed_models: Sequence[str],
    changed_model_files: Sequence[str],
) -> dict[str, Sequence[str]]:
    models = set(name for name in changed_models if name in KNOWN_MODELS)
    models.update(files_to_models(changed_model_files))

    if models:
        return {"model": sorted(models), "dataset": ["synthetic", "synthetic_mixed"]}
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

    changed_models = parse_csv(os.environ.get("CHANGED_MODELS"))
    changed_files = parse_csv(os.environ.get("CHANGED_MODEL_FILES"))

    matrix = choose_matrix(
        args.event_name,
        as_bool(args.full_benchmark),
        changed_models,
        changed_files,
    )
    payload = json.dumps(matrix, separators=(",", ":"))

    if args.output:
        with open(args.output, "a", encoding="utf-8") as fh:
            fh.write(f"matrix={payload}\n")
    else:
        print(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

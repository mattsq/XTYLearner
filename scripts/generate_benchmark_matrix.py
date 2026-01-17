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

# Models that support k=None (continuous treatment)
CONTINUOUS_TREATMENT_MODELS = {
    "bridge_diff",
    "cevae_m",
    "ctm_t",
    "cycle_dual",
    "deconfounder_cfm",
    "ganite",
    "gnn_ebm",
    "gnn_scm",
    "lp_knn",
    "lt_flow_diff",
    "mean_teacher",
    "prob_circuit",
    "scgm",
    "ss_dml",
    "tab_jepa",
}

PR_MATRIX = {
    "model": ["cycle_dual", "mean_teacher"],
    "dataset": ["synthetic", "synthetic_mixed_continuous", "criteo_uplift", "nhefs"],
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
        "ccl_cpc",
        "cevae_m",
        "cnflow",
        "crf",
        "crf_discrete",
        "ctm_t",
        "cycle_vat",
        "dag_transformer",
        "deconfounder_cfm",
        "diffusion_gnn_scm",
        "eg_ddi",
        "em",
        "gflownet_treatment",
        "gnn_ebm",
        "jsbf",
        "lp_knn",
        "lt_flow_diff",
        "m2_vae",
        "scgm",
        "vime",
    ],
    "dataset": ["synthetic", "synthetic_mixed", "synthetic_mixed_continuous", "criteo_uplift", "nhefs"],
}

# Comprehensive matrix for workflow_dispatch: all models that work on GitHub runners
# Excludes only ss_dml which requires optional doubleml dependency
WORKFLOW_DISPATCH_MATRIX = {
    "model": [
        "bridge_diff",
        "cacore",
        "ccl_cpc",
        "cevae_m",
        "cnflow",
        "crf",
        "crf_discrete",
        "ctm_t",
        "cycle_dual",
        "cycle_vat",
        "dag_transformer",
        "deconfounder_cfm",
        "diffusion_cevae",
        "diffusion_gnn_scm",
        "dragon_net",
        "eg_ddi",
        "em",
        "factor_vae_plus",
        "fixmatch",
        "flow_ssc",
        "ganite",
        "gflownet_treatment",
        "gnn_ebm",
        "gnn_scm",
        "joint_ebm",
        "jsbf",
        "lp_knn",
        "lt_flow_diff",
        "m2_vae",
        "masked_tabular_transformer",
        "mean_teacher",
        "multitask",
        "prob_circuit",
        "scgm",
        "semiite",
        "ss_cevae",
        "tab_jepa",
        "vacim",
        "vat",
        "vime",
    ],
    "dataset": ["synthetic", "synthetic_mixed", "synthetic_mixed_continuous", "criteo_uplift", "nhefs"],
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
    "dataset": ["synthetic", "synthetic_mixed", "synthetic_mixed_continuous", "criteo_uplift", "nhefs"],
}


def as_bool(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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

REGISTERED_MODELS = {
    name
    for names in MODULE_MODEL_MAP.values()
    for name in names
}

KNOWN_MODELS = (
    set(FULL_MATRIX["model"])
    | set(DEFAULT_MATRIX["model"])
    | set(PR_MATRIX["model"])
    | set(WORKFLOW_DISPATCH_MATRIX["model"])
    | REGISTERED_MODELS
)


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


def filter_matrix(matrix: dict[str, Sequence[str]]) -> dict:
    """Filter model-dataset combinations to ensure compatibility.

    Specifically, synthetic_mixed_continuous requires models that support k=None.
    Returns matrix with exclude list for incompatible combinations.
    """
    models = matrix.get("model", [])
    datasets = matrix.get("dataset", [])

    # If synthetic_mixed_continuous is not in datasets, no filtering needed
    if "synthetic_mixed_continuous" not in datasets:
        return matrix

    # Build exclude list for models that don't support continuous treatment
    exclude = []
    for model in models:
        if model not in CONTINUOUS_TREATMENT_MODELS:
            exclude.append({"model": model, "dataset": "synthetic_mixed_continuous"})

    # Return matrix with exclude list if there are exclusions
    if exclude:
        return {"model": list(models), "dataset": list(datasets), "exclude": exclude}

    return matrix


def choose_matrix(
    event_name: str,
    full_benchmark: bool,
    changed_models: Sequence[str],
    changed_model_files: Sequence[str],
) -> dict:
    models = set(name for name in changed_models if name in KNOWN_MODELS)
    models.update(files_to_models(changed_model_files))

    if models:
        matrix = {"model": sorted(models), "dataset": ["synthetic", "synthetic_mixed", "synthetic_mixed_continuous", "criteo_uplift", "nhefs"]}
        return filter_matrix(matrix)
    if event_name == "pull_request":
        return filter_matrix(PR_MATRIX)
    if event_name == "workflow_dispatch":
        return filter_matrix(WORKFLOW_DISPATCH_MATRIX)
    if full_benchmark:
        return filter_matrix(FULL_MATRIX)
    return filter_matrix(DEFAULT_MATRIX)


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

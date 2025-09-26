"""Tests for the benchmark matrix generator utility."""

from scripts.generate_benchmark_matrix import (
    REGISTERED_MODELS,
    choose_matrix,
    files_to_models,
)


def test_files_to_models_discovers_newly_registered_model() -> None:
    """Newly added models should be resolved from their source paths."""

    assert "crf" in REGISTERED_MODELS

    resolved = files_to_models(["xtylearner/models/crf.py"])

    assert "crf" in resolved


def test_choose_matrix_filters_to_registered_models() -> None:
    """The generated matrix should only contain recognised model names."""

    matrix = choose_matrix(
        event_name="pull_request",
        full_benchmark=False,
        changed_models=["crf", "does_not_exist"],
        changed_model_files=[],
    )

    assert matrix["model"] == ["crf"]

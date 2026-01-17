import numpy as np
import pandas as pd
import torch
import pytest

from xtylearner.data import (
    load_toy_dataset,
    load_synthetic_dataset,
    load_mixed_synthetic_dataset,
    get_dataset,
)
from xtylearner import load_tabular_dataset


def test_load_toy_dataset_shapes():
    ds = load_toy_dataset(n_samples=10, d_x=3, seed=1)
    X, Y, T = ds.tensors
    assert X.shape == (10, 3)
    assert Y.shape == (10, 1)
    assert T.shape == (10,)


def test_load_synthetic_dataset_shapes():
    ds = load_synthetic_dataset(n_samples=8, d_x=4, seed=2)
    X, Y, T = ds.tensors
    assert X.shape == (8, 4)
    assert Y.shape == (8, 1)
    assert T.shape == (8,)


def test_load_mixed_synthetic_dataset_shapes():
    ds = load_mixed_synthetic_dataset(n_samples=10, d_x=3, seed=3, label_ratio=0.6)
    X, Y, T = ds.tensors
    assert X.shape == (10, 3)
    assert Y.shape == (10, 1)
    assert T.shape == (10,)
    assert (T == -1).sum() > 0
    assert (T >= 0).sum() > 0


def _create_frame(n: int = 5):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n).astype(np.float32),
            "x2": rng.normal(size=n).astype(np.float32),
            "outcome": rng.normal(size=n).astype(np.float32),
            "treatment": rng.integers(0, 2, size=n).astype(np.int64),
        }
    )
    return df


def test_load_tabular_from_dataframe():
    df = _create_frame(6)
    ds = load_tabular_dataset(df)
    X, Y, T = ds.tensors
    assert X.shape == (6, 2)
    assert Y.shape == (6, 1)
    assert T.shape == (6,)


def test_load_tabular_from_csv(tmp_path):
    df = _create_frame(4)
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    ds = load_tabular_dataset(path)
    X, Y, T = ds.tensors
    assert X.shape == (4, 2)
    assert Y.shape == (4, 1)
    assert T.shape == (4,)


def test_load_tabular_from_array():
    df = _create_frame(3)
    arr = df.to_numpy()
    ds = load_tabular_dataset(arr)
    X, Y, T = ds.tensors
    assert X.shape == (3, 2)
    assert Y.shape == (3, 1)
    assert T.shape == (3,)


def test_load_tabular_multiple_outcomes_dataframe():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=5).astype(np.float32),
            "x2": rng.normal(size=5).astype(np.float32),
            "y1": rng.normal(size=5).astype(np.float32),
            "y2": rng.normal(size=5).astype(np.float32),
            "t": rng.integers(0, 2, size=5).astype(np.int64),
        }
    )
    ds = load_tabular_dataset(df, outcome_col=["y1", "y2"], treatment_col="t")
    X, Y, T = ds.tensors
    assert X.shape == (5, 2)
    assert Y.shape == (5, 2)
    assert T.shape == (5,)


def test_load_tabular_multiple_outcomes_array():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 2)).astype(np.float32)
    Y = rng.normal(size=(4, 2)).astype(np.float32)
    T = rng.integers(0, 2, size=(4, 1)).astype(np.int64)
    arr = np.concatenate([X, Y, T], axis=1)
    ds = load_tabular_dataset(arr, outcome_col=2)
    X_t, Y_t, T_t = ds.tensors
    assert X_t.shape == (4, 2)
    assert Y_t.shape == (4, 2)
    assert T_t.shape == (4,)


def test_load_tabular_with_string_treatments():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=4).astype(np.float32),
            "x2": rng.normal(size=4).astype(np.float32),
            "outcome": rng.normal(size=4).astype(np.float32),
            "treatment": ["a", "b", "a", "b"],
        }
    )
    ds = load_tabular_dataset(df)
    X, Y, T = ds.tensors
    assert X.shape == (4, 2)
    assert Y.shape == (4, 1)
    assert T.dtype == torch.int64
    assert getattr(ds, "treatment_mapping") == {"a": 0, "b": 1}


# Additional tests


def test_load_tabular_invalid_type():
    with pytest.raises(TypeError):
        load_tabular_dataset(123)


def test_load_tabular_missing_columns():
    df = pd.DataFrame({"x1": [0.0, 1.0], "x2": [0.0, 1.0]})
    with pytest.raises(ValueError):
        load_tabular_dataset(df)


def test_load_tabular_bad_numpy_shape():
    arr = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        load_tabular_dataset(arr, outcome_col=1)


def test_continuous_treatment_toy_dataset():
    ds = load_toy_dataset(n_samples=4, d_x=2, seed=0, continuous_treatment=True)
    _, _, T = ds.tensors
    assert T.dtype == torch.float32


def test_continuous_treatment_mixed_synthetic_dataset():
    ds = load_mixed_synthetic_dataset(
        n_samples=10, d_x=3, seed=3, label_ratio=0.6, continuous_treatment=True
    )
    X, Y, T = ds.tensors
    assert X.shape == (10, 3)
    assert Y.shape == (10, 1)
    assert T.shape == (10,)
    assert T.dtype == torch.float32
    # Check that some treatments are masked (-1)
    assert (T == -1).sum() > 0
    # Check that some treatments are observed (not -1)
    assert (T != -1).sum() > 0


def test_get_dataset_synthetic_mixed_continuous():
    ds = get_dataset("synthetic_mixed_continuous", n_samples=8, d_x=2, seed=5, label_ratio=0.5)
    X, Y, T = ds.tensors
    assert X.shape == (8, 2)
    assert Y.shape == (8, 1)
    assert T.shape == (8,)
    assert T.dtype == torch.float32
    # Verify it's a mixed dataset (some masked treatments)
    assert (T == -1).sum() > 0
    assert (T != -1).sum() > 0


def test_load_tabular_float_treatment():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=3).astype(np.float32),
            "outcome": rng.normal(size=3).astype(np.float32),
            "treatment": rng.normal(size=3).astype(np.float32),
        }
    )
    ds = load_tabular_dataset(df, outcome_col="outcome", treatment_dtype=float)
    _, _, T = ds.tensors
    assert T.dtype == torch.float32

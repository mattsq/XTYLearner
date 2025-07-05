import numpy as np
import pandas as pd

from xtylearner.data import (
    load_toy_dataset,
    load_synthetic_dataset,
    load_mixed_synthetic_dataset,
    load_tabular_dataset,
)


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

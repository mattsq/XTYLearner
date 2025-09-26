"""Utility functions for the Criteo Uplift dataset.

The loader attempts to fetch the real Criteo Uplift dataset using multiple methods:
1. Direct download from Criteo AI Lab
2. scikit-uplift library (if available)
3. Fallback to synthetic data

The real dataset consists of 25M rows with 11 features, treatment indicator,
and 2 labels (visits and conversions) from incrementality tests in advertising.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path
from typing import Literal
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

# Try importing scikit-uplift
try:
    from sklift.datasets import fetch_criteo
    HAS_SKLIFT = True
except ImportError:
    HAS_SKLIFT = False

# Direct download URL from Criteo
CRITEO_URL = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"


def _load_real_criteo_sklift(sample_frac: float, seed: int, outcome: str):
    """Try to load real Criteo dataset using scikit-uplift."""
    if not HAS_SKLIFT:
        return None

    try:
        print("Attempting to load real Criteo dataset via scikit-uplift...")
        # Try to fetch with smaller sample first
        percent10 = sample_frac <= 0.1
        bunch = fetch_criteo(percent10=percent10, return_X_y_t=False)

        X = bunch.data.astype(np.float32)
        T = bunch.target_treatment.astype(np.int64)

        if outcome == "visit":
            Y = bunch.target_visit.astype(np.float32).reshape(-1, 1)
        else:  # conversion
            Y = bunch.target_conversion.astype(np.float32).reshape(-1, 1)

        # Further sample if needed
        if sample_frac < 1.0 and not percent10:
            n_samples = len(X)
            n_keep = int(n_samples * sample_frac)

            rng = np.random.RandomState(seed)
            indices = rng.choice(n_samples, n_keep, replace=False)

            X = X[indices]
            Y = Y[indices]
            T = T[indices]

        print(f"Successfully loaded real Criteo dataset: {len(X)} samples")
        return X, Y, T

    except Exception as e:
        print(f"scikit-uplift failed: {e}")
        return None


def _load_real_criteo_direct(data_dir: str, sample_frac: float, seed: int, outcome: str):
    """Try to load real Criteo dataset via direct download."""
    try:
        print("Attempting to download real Criteo dataset directly...")

        path = Path(data_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)

        # Download compressed file
        gz_path = path / "criteo-uplift-v2.1.csv.gz"
        csv_path = path / "criteo-uplift-v2.1.csv"

        if not csv_path.exists():
            if not gz_path.exists():
                urlretrieve(CRITEO_URL, gz_path.as_posix())

            # Decompress
            with gzip.open(gz_path, 'rb') as f_in:
                with open(csv_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        # Load with memory-efficient chunked sampling
        if sample_frac < 1.0:
            # Use chunked reading with probabilistic sampling to avoid memory issues
            print(f"Using memory-efficient chunked sampling with {sample_frac*100:.2f}% sample rate...")

            chunk_size = 50000  # Process 50k rows at a time
            sampled_chunks = []
            rng = np.random.RandomState(seed)

            # Read and sample in chunks to avoid loading entire file into memory
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                # Sample rows from this chunk probabilistically
                if len(chunk) > 0:
                    # Use binomial sampling for each chunk
                    n_sample = rng.binomial(len(chunk), sample_frac)
                    if n_sample > 0:
                        sampled_chunk = chunk.sample(n=n_sample, random_state=rng.randint(0, 2**31))
                        sampled_chunks.append(sampled_chunk)

            # Concatenate all sampled chunks
            if sampled_chunks:
                df = pd.concat(sampled_chunks, ignore_index=True)
            else:
                # If no samples, create empty dataframe with correct columns
                df = pd.read_csv(csv_path, nrows=0)
        else:
            df = pd.read_csv(csv_path)

        # Extract features
        feature_cols = [f"f{i}" for i in range(12)]  # f0-f11
        X = df[feature_cols].values.astype(np.float32)
        T = df["treatment"].values.astype(np.int64)
        Y = df[outcome].values.astype(np.float32).reshape(-1, 1)

        print(f"Successfully loaded real Criteo dataset: {len(X)} samples")
        return X, Y, T

    except Exception as e:
        print(f"Direct download failed: {e}")
        return None


def _generate_synthetic_criteo(n_samples: int, seed: int, outcome: str):
    """Generate synthetic Criteo-like dataset."""
    print(f"Generating synthetic Criteo-like dataset...")

    rng = np.random.RandomState(seed)

    # Generate 11 features (similar to real Criteo dataset)
    X = rng.randn(n_samples, 11).astype(np.float32)

    # Generate propensity scores based on features
    propensity_logits = X[:, :3].sum(axis=1) * 0.5  # Use first 3 features
    propensity = 1 / (1 + np.exp(-propensity_logits))

    # Generate treatment assignment
    T = rng.binomial(1, propensity).astype(np.int64)

    # Generate outcome with treatment effect
    # Base outcome depends on features
    base_outcome_logits = X[:, 3:6].sum(axis=1) * 0.3

    # Treatment effect (uplift) - varies by individual
    treatment_effect = X[:, 6:9].sum(axis=1) * 0.2

    # Final outcome
    if outcome == "conversion":
        # Lower base rate for conversions
        outcome_logits = base_outcome_logits - 2.0 + T * treatment_effect
    else:  # visit
        # Higher base rate for visits
        outcome_logits = base_outcome_logits - 0.5 + T * treatment_effect

    outcome_probs = 1 / (1 + np.exp(-outcome_logits))
    Y = rng.binomial(1, outcome_probs).astype(np.float32).reshape(-1, 1)

    return X, Y, T


def load_criteo_uplift(
    split: Literal["train", "test"] = "train",
    *,
    n_samples: int = 10000,
    sample_frac: float = 0.01,
    data_dir: str = "~/.xtylearner/data",
    seed: int = 42,
    outcome: Literal["visit", "conversion"] = "conversion",
    prefer_real: bool = True,
) -> TensorDataset:
    """Load Criteo Uplift dataset (real or synthetic).

    Parameters
    ----------
    split:
        Which portion of the dataset to load, ``"train"`` or ``"test"``.
    n_samples:
        Number of samples to generate (synthetic mode only).
    sample_frac:
        Fraction of the full dataset to sample (real dataset only).
    data_dir:
        Directory where downloaded files are stored.
    seed:
        Random seed.
    outcome:
        Which outcome to use: ``"visit"`` or ``"conversion"``.
    prefer_real:
        If True, try to load real dataset before falling back to synthetic.

    Returns
    -------
    TensorDataset
        TensorDataset with covariates ``X``, outcomes ``Y`` and treatment ``T``.
        X has 11 features, Y is binary outcome, T is binary treatment.
    """

    X, Y, T = None, None, None

    if prefer_real:
        # Try scikit-uplift first
        result = _load_real_criteo_sklift(sample_frac, seed, outcome)
        if result is not None:
            X, Y, T = result
        else:
            # Try direct download
            result = _load_real_criteo_direct(data_dir, sample_frac, seed, outcome)
            if result is not None:
                X, Y, T = result

    # Fallback to synthetic
    if X is None:
        X, Y, T = _generate_synthetic_criteo(n_samples, seed, outcome)

    # Convert to tensors
    X_tensor = torch.from_numpy(X)
    Y_tensor = torch.from_numpy(Y)
    T_tensor = torch.from_numpy(T)

    # Split into train/test
    n_samples = len(X)
    train_size = int(0.8 * n_samples)

    if split == "train":
        X_tensor = X_tensor[:train_size]
        Y_tensor = Y_tensor[:train_size]
        T_tensor = T_tensor[:train_size]
    else:  # test
        X_tensor = X_tensor[train_size:]
        Y_tensor = Y_tensor[train_size:]
        T_tensor = T_tensor[train_size:]

    print(f"Final dataset: {len(X_tensor)} samples")
    print(f"Treatment rate: {T_tensor.float().mean():.3f}")
    print(f"Outcome rate: {Y_tensor.float().mean():.3f}")

    return TensorDataset(X_tensor, Y_tensor, T_tensor)


__all__ = ["load_criteo_uplift"]
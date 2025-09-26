# Criteo Uplift Dataset Integration

The XTYLearner framework now includes integrated support for the Criteo Uplift dataset, a large-scale benchmark dataset for causal inference and uplift modeling.

## Dataset Overview

- **Size**: 25M samples with 11 features
- **Treatment**: Binary (treatment/control)
- **Outcomes**: Binary (visit/conversion)
- **Domain**: Online advertising incrementality tests
- **Paper**: "A Large Scale Benchmark for Uplift Modeling" (AdKDD 2018)

## Usage

```python
from xtylearner.data import get_dataset
from examples.benchmark_models import _run_single

# Load dataset (will try real data first, fallback to synthetic)
ds = get_dataset('criteo_uplift', sample_frac=0.01, prefer_real=True)

# Run benchmark
result = _run_single(('criteo_uplift', 'factor_vae_plus'))
```

## Accessing Real Dataset

The framework tries multiple methods to access the real Criteo dataset:

### Method 1: scikit-uplift (Recommended)

```bash
# Install scikit-uplift in your environment
pip install scikit-uplift

# Then use prefer_real=True (default)
ds = get_dataset('criteo_uplift', sample_frac=0.01, prefer_real=True)
```

### Method 2: Direct Download

The framework automatically tries to download from:
```
http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz
```

Files are cached in `~/.xtylearner/data/` for reuse.

### Method 3: Synthetic Fallback

If real data is unavailable, the framework generates synthetic data with:
- Same structure (11 features, binary treatment/outcome)
- Realistic propensity scores and treatment effects
- Configurable sample size

## Configuration Options

```python
load_criteo_uplift(
    split="train",                    # "train" or "test"
    sample_frac=0.01,                # Fraction of full dataset (real data)
    n_samples=10000,                 # Sample count (synthetic data)
    data_dir="~/.xtylearner/data",   # Download directory
    seed=42,                         # Random seed
    outcome="conversion",            # "visit" or "conversion"
    prefer_real=True,                # Try real data first
)
```

## Benchmark Integration

The dataset is fully integrated with the automated benchmark workflow:

```python
from examples.benchmark_models import _run_single

# Test any model on Criteo dataset
models = ['factor_vae_plus', 'ganite', 'prob_circuit', 'mean_teacher']

for model in models:
    result = _run_single(('criteo_uplift', model))
    print(f"{model}: accuracy={result['val treatment accuracy']:.3f}, "
          f"RMSE={result['val outcome rmse']:.3f}")
```

## Performance Example

| Model | Treatment Accuracy | Outcome RMSE |
|-------|-------------------|---------------|
| factor_vae_plus | 0.473 | 1.410 |
| prob_circuit | 0.645 | 0.338 |
| ganite | 0.650 | 0.385 |
| mean_teacher | 0.633 | 0.404 |

## Troubleshooting

**Issue**: Real dataset download fails
**Solution**: Set `prefer_real=False` to use synthetic fallback

**Issue**: scikit-uplift import error
**Solution**: Install with `pip install scikit-uplift`

**Issue**: Download timeout
**Solution**: The 25M dataset is large (~1GB). Use `sample_frac=0.001` for testing

## Implementation Details

- **Graceful fallback**: Automatically tries real → synthetic data sources
- **Memory-efficient sampling**: Uses chunked reading with probabilistic sampling to avoid loading 25M rows into memory
- **Streaming approach**: Processes data in 50k row chunks, samples from each chunk independently
- **Memory management**: Supports fractional sampling without materializing large index arrays
- **Caching**: Downloads are cached locally for reuse
- **API consistency**: Same TensorDataset(X, Y, T) format as other datasets
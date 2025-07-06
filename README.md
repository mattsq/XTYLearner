# XTYLearner

Flexible implementations of various learners for joint learning of pre-treatment,
treatment and post-treatment outcomes.

## Installation

XTYLearner requires Python 3.8 or newer. The package can be installed directly
from source or, once published, from the Python Package Index.

```bash
# install from a local clone
git clone <repository-url>
cd XTYLearner
pip install -e .

# or install from PyPI when available
pip install xtylearner
```

## API Overview

### Loading Datasets

Datasets are provided via :mod:`xtylearner.data`.  The convenience function
:func:`xtylearner.data.get_dataset` returns a ``torch.utils.data.Dataset`` by
name:

```python
from xtylearner.data import get_dataset

dataset = get_dataset("toy", n_samples=100, d_x=2)
```

Individual loaders such as ``load_toy_dataset`` and ``load_synthetic_dataset``
are also exported for direct use.

``load_tabular_dataset`` converts generic tabular data into a
``torch.utils.data.TensorDataset``.  The input may be a ``pandas.DataFrame``, a
path to a CSV file or a ``numpy.ndarray`` with columns ``X``/``Y``/``T``.  For
data frames and CSV files the names of the outcome and treatment columns can be
specified via ``outcome_col`` and ``treatment_col`` (defaulting to ``"outcome"``
and ``"treatment"``).  When a NumPy array is provided ``outcome_col`` indicates
how many outcome columns appear immediately before the treatment column.  All
remaining columns are treated as covariates.

### Creating Models

Model architectures register themselves with the module
:mod:`xtylearner.models`.  Instantiate a model through the registry using
:func:`xtylearner.models.get_model`:

```python
from xtylearner.models import get_model

model = get_model("m2_vae", d_x=2, d_y=1, k=2)
```

### Available Models

XTYLearner includes a small collection of reference architectures:

- ``CycleDual`` – a cycle-consistent dual network that imputes missing
  treatment labels.
- ``MixtureOfFlows`` – a semi-supervised conditional normalising flow
  combining an invertible network with a classifier.
- ``MultiTask`` – a discriminative network trained with self-training on
  pseudo-labels for partially observed treatments.
- ``M2VAE`` – a generative model based on the M2 variational autoencoder.
- ``SS_CEVAE`` – a semi-supervised extension of the CEVAE framework.
- ``JSBF`` – a score-based diffusion model of the full joint
  distribution ``(X, T, Y)`` supporting missing treatment labels.
- ``BridgeDiff`` – a diffusion bridge architecture that couples
  counterfactual draws ``Y(0)`` and ``Y(1)`` even when treatment labels
  are missing.
- ``LTFlowDiff`` – combines a conditional normalising flow with a
  treatment-conditioned latent diffusion prior.
- ``EnergyDiffusionImputer`` – an energy-guided discrete diffusion model
  for imputing missing treatments.

Each model exposes a ``loss`` method compatible with the trainer utilities
described below.

### Training Utilities

Training utilities live in :mod:`xtylearner.training`.  The high level `Trainer` class automatically selects the appropriate implementation (supervised, generative or diffusion) based on the supplied model.  A typical workflow is:

```python
from torch.utils.data import DataLoader
from xtylearner.training import Trainer

loader = DataLoader(dataset, batch_size=32, shuffle=True)
optimizer = torch.optim.Adam(model.parameters())
trainer = Trainer(model, optimizer, loader)
trainer.fit(5)
loss = trainer.evaluate(loader)
```
An optional learning rate scheduler can be supplied via the ``scheduler``
argument and is stepped after each epoch.

### Handling Missing Treatment Labels

Datasets may contain unobserved treatments denoted by ``-1`` in the
``T`` tensor (see :func:`xtylearner.data.load_mixed_synthetic_dataset`).
All trainers automatically separate labelled and unlabelled rows based on
this value.  When only ``(X, Y)`` pairs are provided, the trainer will
internally set ``T`` to ``-1`` for every sample.

Models such as ``CycleDual`` and ``MixtureOfFlows`` impute or marginalise over
the missing labels, while generative models use a
semi-supervised objective using both labelled and unlabelled data.  A typical
workflow is:

```python
from xtylearner.data import load_mixed_synthetic_dataset
from xtylearner.training import Trainer

dataset = load_mixed_synthetic_dataset(n_samples=100, label_ratio=0.3)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = get_model("m2_vae", d_x=2, d_y=1, k=2)
optimizer = torch.optim.Adam(model.parameters())
trainer = Trainer(model, optimizer, loader)
trainer.fit(5)
```

For score-based diffusion models like ``JSBF`` the `Trainer`
handles the optimisation:

```python
from xtylearner.models import JSBF
from xtylearner.training import Trainer

model = JSBF(d_x=2, d_y=1)
trainer = Trainer(model, optimizer, loader)
trainer.fit(5)
```

## Command Line Interface

The package exposes a simple CLI for training defined in
``xtylearner.scripts.train``.  After installation the entry point
``xtylearner-train`` becomes available:

```bash
xtylearner-train --model m2_vae --dataset toy
```

This command loads the default configuration from
``xtylearner/configs/default.yaml``.  A custom YAML config file may be supplied
with ``--config`` to override any settings.

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

# enable causal baselines
pip install xtylearner[causal]
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

Example usage with a ``DataLoader``:

```python
from xtylearner import load_tabular_dataset
from torch.utils.data import DataLoader

dataset = load_tabular_dataset("data.csv")
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Creating Models

Model architectures register themselves with the module
:mod:`xtylearner.models`.  Instantiate a model through the registry using
:func:`xtylearner.models.get_model`:

```python
from xtylearner.models import get_model

model = get_model("m2_vae", d_x=2, d_y=1, k=2)
```

### Available Models

The model registry exposes a variety of architectures grouped below by type.

#### Discriminative

- ``CycleDual`` – a cycle-consistent dual network that imputes missing
  treatment labels.
- ``MultiTask`` – a discriminative network trained with self-training on
  pseudo-labels for partially observed treatments.
- ``DragonNet`` – shared encoder with outcome and propensity heads plus
  targeted regularisation for robust effect estimates.
- ``CaCoRE`` – a contrastive representation encoder linking ``h(x)`` with the
  joint outcome-treatment space.
- ``MaskedTabularTransformer`` – a transformer encoder for tabular data using a
  masked-token training objective.
- ``SemiITE`` – a co-training network for semi-supervised treatment effect
  estimation with a reconstruction head for ``p(t\|x,y)``.
- ``CCL_CPCModel`` – a contrastive predictive coding model for sequential
  covariates and partially observed labels.

#### Generative

- ``MixtureOfFlows`` – a semi-supervised conditional normalising flow
  combining an invertible network with a classifier.
- ``CNFlowModel`` – a conditional normalising flow modelling
  ``p(Y\mid X,T)``.
- ``M2VAE`` – a generative model based on the M2 variational autoencoder.
- ``SS_CEVAE`` – a semi-supervised extension of the CEVAE framework.
- ``CEVAE_M`` – CEVAE with latent treatment for partially-observed labels.
- ``JointEBM`` – an energy-based model of the joint distribution ``(X, T, Y)``
  optimised with a contrastive objective.
- ``ProbCircuitModel`` – a probabilistic circuit baseline leveraging SPFlow
  sum-product networks.
- ``GFlowNetTreatment`` – a Generative Flow Network that samples treatments in
  proportion to outcome likelihood.
- ``EMModel`` – a lightweight EM algorithm implementation for linear models.
- ``GANITE`` – a GAN-based approach for individual treatment effect estimation.
- ``VACIM`` – a CEVAE variant with conditional masking for partial encoders.
- ``GNN_SCM`` – a graph neural structural causal model learning latent graphs.
- ``GNN_EBM`` – an energy-based variant of the GNN-SCM model.
- ``FactorVAEPlus`` – a latent variable model with multiple categorical
  treatments.
- ``SCGM`` – a semi-supervised causal generative model.

#### Causal Factor Models

- ``DeconfounderCFM`` – two-stage factor model that learns a substitute
  confounder from multiple correlated treatments.

#### Diffusion

- ``JSBF`` – a score-based diffusion model of the full joint
  distribution ``(X, T, Y)`` supporting missing treatment labels.
- ``BridgeDiff`` – a diffusion bridge architecture that couples
  counterfactual draws ``Y(0)`` and ``Y(1)`` even when treatment labels
  are missing.
- ``LTFlowDiff`` – combines a conditional normalising flow with a
  treatment-conditioned latent diffusion prior.
- ``EnergyDiffusionImputer`` – an energy-guided discrete diffusion model
  for imputing missing treatments.
- ``DiffusionCEVAE`` – a diffusion-based variant of CEVAE trained via latent
  score matching.
- ``DiffusionGNN_SCM`` – a diffusion-based variant of the GNN_SCM model.
- ``CTMT`` – a consistency-trajectory diffusion model with a treatment head.

#### Semi-Supervised Baselines

- ``LP_KNN`` – a k-nearest neighbour label propagation baseline that can
  optionally train a scikit-learn regressor on the propagated labels.
- ``MeanTeacher`` – an exponential moving average teacher model for
  consistency training.
- ``VIME`` – a two-stage self-supervised method for tabular data.
- ``VAT_Model`` – virtual adversarial training for smooth predictions.
- ``FixMatch`` – pseudo-labelling combined with strong data augmentations.
- ``SSDMLModel`` – semi-supervised double machine learning (requires the
  ``xtylearner[causal]`` extra).

Each model exposes a ``loss`` method compatible with the trainer utilities
described below.

### Training Utilities

Training utilities live in :mod:`xtylearner.training`.  The high level `Trainer` class automatically selects the appropriate implementation (supervised, generative or diffusion) based on the supplied model.  A typical workflow is:

```python
import torch
from torch.utils.data import DataLoader
from xtylearner.training import Trainer, ConsoleLogger

loader = DataLoader(dataset, batch_size=32, shuffle=True)
optimizer = torch.optim.Adam(model.parameters())
trainer = Trainer(model, optimizer, loader, logger=ConsoleLogger())
trainer.fit(5)
loss = trainer.evaluate(loader)
```
An optional learning rate scheduler can be supplied via the ``scheduler``
argument and is stepped after each epoch.
Training progress can be logged by passing a ``TrainerLogger`` instance such as
``ConsoleLogger`` when constructing the trainer.

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
import torch
from torch.utils.data import DataLoader
from xtylearner.data import load_mixed_synthetic_dataset
from xtylearner.models import get_model
from xtylearner.training import Trainer

dataset = load_mixed_synthetic_dataset(n_samples=100, d_x=5, label_ratio=0.3)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = get_model("m2_vae", d_x=5, d_y=1, k=2)
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

### Active Learning

Utilities in :mod:`xtylearner.active` implement common query strategies for
selecting the most informative unlabelled points:

- ``EntropyT`` – rank samples by the entropy of the model's treatment
  predictions.
- ``DeltaCATE`` – prioritise points with high variance in predicted treatment
  effects.
- ``FCCMRadius`` – a weighted combination of entropy, effect variance and the
  coverage radius around labelled data.

The :class:`xtylearner.training.ActiveTrainer` wraps a standard trainer with an
active learning loop.  Given a ``TensorDataset`` containing labelled and
unlabelled rows (with ``T = -1`` for missing labels), it repeatedly fits the
model, scores the unlabelled pool with the chosen strategy and moves the top
scoring samples into the labelled set until the budget is exhausted.

```python
from torch.utils.data import DataLoader, TensorDataset
from xtylearner.active import EntropyT
from xtylearner.training import ActiveTrainer

dataset = TensorDataset(X, Y, T)  # use -1 for unlabelled treatments
loader = DataLoader(dataset, batch_size=32, shuffle=True)
trainer = ActiveTrainer(model, optimizer, loader, strategy=EntropyT(),
                        budget=50, batch=10)
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

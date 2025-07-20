# Model development guidelines

## Adding a new model

1. Place the implementation under `xtylearner/models/`.
2. Decorate the class with `@register_model("your_name")` from
   `xtylearner.models.registry` so that it becomes available through
   `get_model`.
3. Import the class in `xtylearner/models/__init__.py` and add its name to
   `__all__`.

## Minimal interfaces

- **Supervised/discriminative models** (e.g. `CycleDual`, `JointEBM`)
  - Subclass `torch.nn.Module`.
  - Provide a `loss(x, y, t_obs)` method returning a scalar tensor.
  - Implement `forward` if predictions depend on treatment.
- Optionally expose `predict_treatment_proba(x, y)` and
    `predict_outcome(x, t)` for evaluation.

  `dragon_net` follows this template. It uses a shared MLP encoder feeding an
  outcome head and a propensity head. A third reconstruction head predicts the
  treatment from both the representation and the observed outcome so that rows
  without labels still inform the network.

  The loss combines mean-squared error for outcomes with cross-entropy for the
  two treatment heads, a KL term encouraging agreement when `T` is missing and a
  targeted regularisation penalty. This mirrors the DragonNet objective of Shi
  et al. (2019).

- **Generative models** (e.g. `DiffusionCEVAE`, `M2VAE`)
  - Subclass `torch.nn.Module`.
  - Implement an `elbo(x, y, t)` or `loss(x, y, t)` method used during training.
  - Include a `k` attribute for the number of treatment classes.
  - Optional helper methods as above.

- **Diffusion/score-based models** (e.g. `JSBF`)
  - Subclass `torch.nn.Module`.
  - Implement `loss(x, y, t)` and provide a sampler via `sample(n)` or
    `paired_sample(n)`.

- **Array-based/EM style models** (e.g. `EMModel`)
  - Implement `fit(X, Y, T)` operating on `numpy.ndarray` inputs.
  - Provide `predict_treatment_proba(Z)` and `predict_outcome(X, t)`.
  - `ss_dml` – Semi-Supervised Double ML

The high level `Trainer` class in `xtylearner/training` automatically picks the
appropriate trainer based on the presence of these methods.

## ccl_cpc

Identifier: `ccl_cpc`

Hyper-parameters:
- `hidden` – encoder width (default 128)
- `lambda_cpc` – weight for the CPC objective
- `lambda_y` – labelled outcome loss weight
- `lambda_t` – labelled treatment loss weight
- `temperature` – InfoNCE softmax temperature
- `k_future` – prediction horizon in steps
- `seq_len` – input sequence length

The **deconfounder** or Causal-Factor Model fits a latent variable model to the set of treatments only. 
After learning a substitute confounder ``z`` it trains a separate outcome network conditioned on ``[x, t, z]``.

### Assumptions

- Hidden confounders affect at least two causes (no single‑cause confounders).
- The treatment factor model is well specified. A posterior predictive check using HSIC between generated treatments and ``z`` should be near zero.

### Key hyper-parameters

- ``d_z`` – dimensionality of the latent confounder.
- ``hidden`` – width of hidden layers in both networks.
- ``pretrain_epochs`` – number of epochs to train the treatment VAE before updating the outcome head.
- ``ppc_freq`` – frequency of HSIC diagnostics during training.

The deconfounder is useful when multiple causes share latent confounders and some treatment values may be missing.

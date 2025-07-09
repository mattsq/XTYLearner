# Diffusion-GNN-SCM

The Diffusion‑GNN‑SCM combines a score based diffusion model with a
learned graph structure.  Covariates are treated as graph roots while
the treatment and outcome nodes exchange messages along a masked
adjacency matrix.  During training Gaussian noise is added following a
cosine schedule and the score network learns to denoise it.

## Objective

The loss mixes diffusion score matching with supervised likelihood
terms.  The adjacency matrix is constrained via the NOTEARS penalty and
L1 sparsity.  Edges from `Y` to any `X_i` are disallowed by the mask.

## Usage tips

* `lambda_acyc` controls the acyclicity penalty.
* `gamma_l1` sets the L1 weight promoting sparsity.


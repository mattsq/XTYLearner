# GNN-EBM

The GNN‑EBM combines a learnable directed graph with an energy-based model of the joint distribution.
A single scalar energy represents the compatibility of ``(x,t,y)`` and message passing enforces global consistency.

## Objective

Training minimises a contrastive divergence loss.  Positive samples use the observed treatments while negative samples are obtained via Langevin dynamics on ``(t,y)``.
An acyclicity penalty ``h(A)=\operatorname{tr}(e^{A\circ A})-d`` keeps the learned adjacency matrix a DAG and L1 regularisation encourages sparsity.

## Usage tips

* ``lambda_acyc`` controls the strength of the acyclicity penalty.
* ``gamma_l1`` sets the L1 weight on the adjacency matrix.
* ``k_langevin`` and ``eta`` tune the Langevin sampler used for negative samples and imputing missing treatments.


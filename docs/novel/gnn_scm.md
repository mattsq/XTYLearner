# GNN-SCM

The Graph-Neural Structural Causal Model (GNN‑SCM) learns a directed acyclic
graph together with neural structural equations.
It provides an exact joint density
\(p_\theta(x,t,y)\) and supports discrete or continuous treatments with
missing labels.

## Objective

Each batch maximises the joint log likelihood of observed variables and
penalises violations of the NOTEARS acyclicity constraint
\(h(A)=\operatorname{tr}(e^{A\circ A})-d\).  A sparsity penalty encourages
simpler graphs.

## Usage tips

* `lambda_acyc` controls the strength of the acyclicity penalty.
* `gamma_l1` sets the L1 regularisation weight on the adjacency matrix.
* Set `forbid_y_to_x=True` (default) to exclude edges from `Y` to any `X_i`.
* Use `predict_treatment_params` to obtain either discrete probabilities or
  the mean and variance of the continuous treatment distribution.

## References

Zheng et al., "DAGs with NO TEARS: Continuous Optimization for Structure
Learning." NeurIPS 2018.

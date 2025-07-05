# Initial pass Train only on the truly-labelled rows (because the rest have T = −1).
# Pseudo-label Run the treatment head on every unlabeled sample;
# if the highest softmax ≥ thr (default 0.9) accept it as a pseudo label.
# Re-train Add accepted pseudo-labels to the labelled pool and continue training.
# Repeat Do n_outer rounds until no new pseudo labels are accepted or a fixed budget is hit.

# Caruana “Multitask Learning”, ML 1997
# people.eecs.berkeley.edu
# Lee “Pseudo-Label: A Simple and Efficient Semi-Supervised Learning Method”, ICML Workshop 2013
# researchgate.net
# Xie et al. “Noisy Student: Self-Training Improves ImageNet Classification”, CVPR 2020
# arxiv.org
# Sohn et al. “FixMatch: Consistency + Confidence for SSL”, NeurIPS 2020
# arxiv.org

import torch
import torch.nn as nn

from .registry import register_model


# ------------------------------------------------------------
# shared encoder for any vector input -------------------------
def mlp(in_dim, out_dim, hidden=128):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


@register_model("multitask")
class MultiTask(nn.Module):
    """
    h               : shared encoder from X  → ℝ^h
    f_Y(h(X),T)     : outcome regressor
    f_T(h(X),Y)     : treatment classifier
    f_X(Y,T)        : inverse regressor
    """

    def __init__(self, d_x, d_y, k, h_dim=128):
        super().__init__()
        self.k = k
        self.h = mlp(d_x, h_dim)

        self.head_Y = mlp(h_dim + k, d_y)  # predict Y
        self.head_T = mlp(d_x + d_y, k)  # predict T
        self.head_X = mlp(d_y + k, d_x)  # reconstruct X

    # --------------------------------------------------------
    def forward(self, X, Y, T_onehot):
        h_x = self.h(X)
        Y_hat = self.head_Y(torch.cat([h_x, T_onehot], -1))
        logits_T = self.head_T(torch.cat([X, Y], -1))
        X_hat = self.head_X(torch.cat([Y, T_onehot], -1))
        return Y_hat, logits_T, X_hat


class DataWrapper(torch.utils.data.Dataset):
    def __init__(self, X, Y, T):
        self.X, self.Y, self.T = X, Y, T  # tensors

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.T[i]

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

from .layers import make_mlp
from .registry import register_model
from ..training.metrics import cross_entropy_loss, mse_loss
from .heads import OrdinalHead
from ..losses import coral_loss, cumulative_link_loss


# ------------------------------------------------------------
# shared encoder for any vector input -------------------------


@register_model("multitask")
class MultiTask(nn.Module):
    """
    h               : shared encoder from X  → ℝ^h
    f_Y(h(X),T)     : outcome regressor
    f_T(h(X),Y)     : treatment classifier
    f_X(Y,T)        : inverse regressor
    """

    def __init__(
        self,
        d_x,
        d_y,
        k,
        h_dim=128,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
        ordinal: bool = False,
        ordinal_method: str = "coral",
    ):
        super().__init__()
        self.k = k
        self.ordinal = ordinal
        self.ordinal_method = ordinal_method
        self.h = make_mlp(
            [d_x, *hidden_dims, h_dim],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

        self.head_Y = make_mlp(
            [h_dim + k, *hidden_dims, d_y],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )  # predict Y

        # Split head_T into features and classification head for ordinal support
        self.head_T_features = make_mlp(
            [d_x + d_y, *hidden_dims],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        t_in_dim = hidden_dims[-1] if hidden_dims else d_x + d_y
        if ordinal:
            self.head_T_classifier = OrdinalHead(t_in_dim, k, method=ordinal_method)
        else:
            self.head_T_classifier = nn.Linear(t_in_dim, k)

        self.head_X = make_mlp(
            [d_y + k, *hidden_dims, d_x],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )  # reconstruct X

    # --------------------------------------------------------
    def _forward_all(self, X, Y, T_onehot):
        h_x = self.h(X)
        Y_hat = self.head_Y(torch.cat([h_x, T_onehot], -1))
        t_features = self.head_T_features(torch.cat([X, Y], -1))
        logits_T = self.head_T_classifier(t_features)
        X_hat = self.head_X(torch.cat([Y, T_onehot], -1))
        return Y_hat, logits_T, X_hat

    # --------------------------------------------------------
    def forward(self, X: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Predict outcome given covariates and treatment."""

        T_1h = torch.nn.functional.one_hot(T.to(torch.long), self.k).float()
        h_x = self.h(X)
        return self.head_Y(torch.cat([h_x, T_1h], dim=-1))

    @torch.no_grad()
    def predict_outcome(self, X: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Wrapper around :meth:`forward` for API consistency."""

        return self.forward(X, T)

    def loss(self, X, Y, T_obs):
        """Supervised loss that ignores missing treatment labels."""

        labelled = T_obs >= 0

        # Replace missing labels by a dummy value for the forward pass
        T_use = torch.where(labelled, T_obs, torch.zeros_like(T_obs))
        T_1h = torch.nn.functional.one_hot(T_use, self.k).float()

        Y_hat, logits_T, X_hat = self._forward_all(X, Y, T_1h)
        if labelled.any():
            loss = mse_loss(Y_hat[labelled], Y[labelled])
            loss += mse_loss(X_hat[labelled], X[labelled])
            # Use ordinal losses if ordinal mode is enabled
            if self.ordinal:
                if self.ordinal_method == "coral":
                    loss += coral_loss(logits_T[labelled], T_obs[labelled].to(torch.long), self.k)
                elif self.ordinal_method == "cumulative":
                    loss += cumulative_link_loss(logits_T[labelled], T_obs[labelled].to(torch.long), self.k)
                else:
                    loss += cross_entropy_loss(logits_T[labelled], T_obs[labelled])
            else:
                loss += cross_entropy_loss(logits_T[labelled], T_obs[labelled])
        else:
            # ensure returned tensor participates in autograd graph
            loss = (Y_hat.sum() + logits_T.sum() + X_hat.sum()) * 0.0
        return loss

    # --------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Return posterior ``p(t|x,y)`` from the classification head."""

        t_features = self.head_T_features(torch.cat([X, Y], dim=-1))
        if self.ordinal and self.ordinal_method in ["coral", "cumulative"]:
            return self.head_T_classifier.predict_proba(t_features)
        return self.head_T_classifier(t_features).softmax(dim=-1)


class DataWrapper(torch.utils.data.Dataset):
    def __init__(self, X, Y, T):
        """Simple container exposing tensors as a ``Dataset``."""

        self.X, self.Y, self.T = X, Y, T  # tensors

    def __len__(self):
        """Dataset size."""

        return len(self.X)

    def __getitem__(self, i):
        """Return the ``i``-th triplet ``(X, Y, T)``."""

        return self.X[i], self.Y[i], self.T[i]


__all__ = ["MultiTask", "DataWrapper"]

# Zhu et al. “Unpaired Image-to-Image Translation using CycleGAN”, ICCV 2017
# arxiv.org

# Yi et al. “DualGAN: Unsupervised Dual Learning for Image-to-Image Translation”, ICCV 2017
# arxiv.org


import torch
import torch.nn as nn
from torch.nn.functional import one_hot

from .layers import make_mlp
from .heads import LowRankDiagHead
from ..losses import nll_lowrank_diag

from .registry import register_model
from ..training.metrics import cross_entropy_loss, mse_loss


# ---------- dual-network module -----------------------------------------
@register_model("cycle_dual")
class CycleDual(nn.Module):
    """
    G_Y : (X ⊕ onehot(T)) → Ŷ
    G_X : (onehot(T) ⊕ Y) → X̂
    C   : (X ⊕ Y) → logits(T)
    """

    def __init__(
        self,
        d_x,
        d_y,
        k,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
        lowrank_head: bool = False,
        rank: int = 4,
    ):
        super().__init__()
        self.k = k
        self.lowrank_head = lowrank_head
        if lowrank_head:
            self.G_Y = make_mlp(
                [d_x + k, *hidden_dims],
                activation=activation,
                dropout=dropout,
                norm_layer=norm_layer,
            )
            in_dim = hidden_dims[-1] if hidden_dims else d_x + k
            self.G_Y_head = LowRankDiagHead(in_dim, d_y, rank)
        else:
            self.G_Y = make_mlp(
                [d_x + k, *hidden_dims, d_y],
                activation=activation,
                dropout=dropout,
                norm_layer=norm_layer,
            )
            self.G_Y_head = None
        self.G_X = make_mlp(
            [k + d_y, *hidden_dims, d_x],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.C = make_mlp(
            [d_x + d_y, *hidden_dims, k],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    # ------------------------------------------------------------------
    def forward(self, X: torch.Tensor, T: torch.Tensor):
        """Predict outcome ``Y`` from covariates ``X`` and treatment ``T``."""

        T_1h = one_hot(T.to(torch.long), self.k).float()
        h = self.G_Y(torch.cat([X, T_1h], dim=-1))
        if self.lowrank_head:
            return self.G_Y_head(h)
        else:
            return h

    @torch.no_grad()
    def predict_outcome(self, X: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Wrapper around :meth:`forward` for API consistency."""

        out = self.forward(X, T)
        if self.lowrank_head:
            return out[0]
        else:
            return out

    # ------------------------------------------------------------------
    def loss(self, X, Y, T_obs, λ_sup=1.0, λ_cyc=1.0, λ_ent=0.1):
        """
        X : (B, d_x)     Y : (B, d_y)
        T_obs : (B,) int in [0,K-1] if labelled, −1 if unknown
        """
        labelled = T_obs >= 0
        unlabelled = ~labelled

        # === STEP 1 : make *some* treatment for every row =============
        logits_T = self.C(torch.cat([X, Y], -1))  # (B,K)
        T_pred = logits_T.argmax(-1)
        T_use = torch.where(labelled, T_obs, T_pred)  # int64
        T_1h = one_hot(T_use, self.k).float()  # (B,K)

        # === STEP 2 : forward + backward generators ===================
        if self.lowrank_head:
            h_y = self.G_Y(torch.cat([X, T_1h], -1))
            mu, Fmat, sigma2 = self.G_Y_head(h_y)
            Y_hat = mu
        else:
            Y_hat = self.G_Y(torch.cat([X, T_1h], -1))
            mu = Fmat = sigma2 = None
        X_hat = self.G_X(torch.cat([T_1h, Y], -1))

        # cycles
        X_cyc = self.G_X(torch.cat([T_1h, Y_hat.detach()], -1))
        if self.lowrank_head:
            h_cyc = self.G_Y(torch.cat([X_hat.detach(), T_1h], -1))
            mu_cyc, F_cyc, sigma2_cyc = self.G_Y_head(h_cyc)
            Y_cyc = mu_cyc
        else:
            Y_cyc = self.G_Y(torch.cat([X_hat.detach(), T_1h], -1))

        # === STEP 3 : losses ==========================================
        mse = mse_loss

        # supervised parts (labelled rows only)
        if self.lowrank_head:
            L_sup_Y = (
                nll_lowrank_diag(
                    Y[labelled], mu[labelled], Fmat[labelled], sigma2[labelled]
                ).mean()
                if labelled.any()
                else 0.0
            )
        else:
            L_sup_Y = mse(Y_hat[labelled], Y[labelled]) if labelled.any() else 0.0
        L_sup_X = mse(X_hat[labelled], X[labelled]) if labelled.any() else 0.0
        L_sup_T = (
            cross_entropy_loss(logits_T[labelled], T_obs[labelled])
            if labelled.any()
            else 0.0
        )

        # unsupervised recon on ALL rows (helps stabilise)
        if self.lowrank_head:
            L_rec_Y = nll_lowrank_diag(Y, mu, Fmat, sigma2).mean()
        else:
            L_rec_Y = mse(Y_hat, Y)
        L_rec_X = mse(X_hat, X)

        # cycle consistency  (ALL rows)
        L_cyc_X = mse(X_cyc, X)
        if self.lowrank_head:
            L_cyc_Y = nll_lowrank_diag(Y, mu_cyc, F_cyc, sigma2_cyc).mean()
        else:
            L_cyc_Y = mse(Y_cyc, Y)

        # entropy regulariser – push classifier to be confident on unlabelled
        if unlabelled.any():
            P_ulb = logits_T[unlabelled].softmax(-1)
            L_ent = -(P_ulb * P_ulb.log()).sum(-1).mean()
        else:
            L_ent = 0.0

        # total
        L_total = (
            λ_sup * (L_sup_X + L_sup_Y + L_sup_T)
            + L_rec_X
            + L_rec_Y
            + λ_cyc * (L_cyc_X + L_cyc_Y)
            + λ_ent * L_ent
        )
        return L_total

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Return posterior ``p(t|x,y)`` from the classifier ``C``."""

        logits = self.C(torch.cat([X, Y], dim=-1))
        return logits.softmax(dim=-1)


__all__ = ["CycleDual"]

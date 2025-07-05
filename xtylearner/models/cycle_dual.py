# Zhu et al. “Unpaired Image-to-Image Translation using CycleGAN”, ICCV 2017
# arxiv.org

# Yi et al. “DualGAN: Unsupervised Dual Learning for Image-to-Image Translation”, ICCV 2017
# arxiv.org


import torch
import torch.nn as nn
from torch.nn.functional import one_hot, cross_entropy

from .layers import make_mlp

from .registry import register_model


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
    ):
        super().__init__()
        self.k = k
        self.G_Y = make_mlp(
            [d_x + k, *hidden_dims, d_y],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
        )
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
        Y_hat = self.G_Y(torch.cat([X, T_1h], -1))
        X_hat = self.G_X(torch.cat([T_1h, Y], -1))

        # cycles
        X_cyc = self.G_X(torch.cat([T_1h, Y_hat.detach()], -1))
        Y_cyc = self.G_Y(torch.cat([X_hat.detach(), T_1h], -1))

        # === STEP 3 : losses ==========================================
        mse = nn.functional.mse_loss

        # supervised parts (labelled rows only)
        L_sup_Y = mse(Y_hat[labelled], Y[labelled]) if labelled.any() else 0.0
        L_sup_X = mse(X_hat[labelled], X[labelled]) if labelled.any() else 0.0
        L_sup_T = (
            cross_entropy(logits_T[labelled], T_obs[labelled])
            if labelled.any()
            else 0.0
        )

        # unsupervised recon on ALL rows (helps stabilise)
        L_rec_Y = mse(Y_hat, Y)
        L_rec_X = mse(X_hat, X)

        # cycle consistency  (ALL rows)
        L_cyc_X = mse(X_cyc, X)
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

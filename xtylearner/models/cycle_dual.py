# Zhu et al. “Unpaired Image-to-Image Translation using CycleGAN”, ICCV 2017
# arxiv.org

# Yi et al. “DualGAN: Unsupervised Dual Learning for Image-to-Image Translation”, ICCV 2017
# arxiv.org


import torch
import torch.nn as nn

from .layers import make_mlp
from .heads import LowRankDiagHead, OrdinalHead
from ..losses import nll_lowrank_diag, coral_loss, cumulative_link_loss

from .registry import register_model
from ..training.metrics import cross_entropy_loss, mse_loss


# ---------- dual-network module -----------------------------------------
@register_model("cycle_dual")
class CycleDual(nn.Module):
    """
    G_Y : (X ⊕ onehot(T)) → Ŷ
    G_X : (onehot(T) ⊕ Y) → X̂
    C   : (X ⊕ Y) → logits(T) or regression output
    """

    def __init__(
        self,
        d_x,
        d_y,
        k: int | None = None,
        *,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout=None,
        norm_layer=None,
        residual: bool = False,
        lowrank_head: bool = False,
        rank: int = 4,
        ordinal: bool = False,
        ordinal_method: str = "coral",
    ):
        super().__init__()
        self.k = k
        self.lowrank_head = lowrank_head
        self.ordinal = ordinal
        self.ordinal_method = ordinal_method
        d_t = k if k is not None else 1

        if k is not None:
            self.t_embedding = nn.Embedding.from_pretrained(
                torch.eye(k), freeze=True
            )
        else:
            self.t_embedding = None

        if lowrank_head:
            self.G_Y = make_mlp(
                [d_x + d_t, *hidden_dims],
                activation=activation,
                dropout=dropout,
                norm_layer=norm_layer,
                residual=residual,
            )
            in_dim = hidden_dims[-1] if hidden_dims else d_x + d_t
            self.G_Y_head = LowRankDiagHead(in_dim, d_y, rank)
        else:
            self.G_Y = make_mlp(
                [d_x + d_t, *hidden_dims, d_y],
                activation=activation,
                dropout=dropout,
                norm_layer=norm_layer,
                residual=residual,
            )
            self.G_Y_head = None

        self.G_X = make_mlp(
            [d_t + d_y, *hidden_dims, d_x],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer,
            residual=residual,
        )

        # Classifier C: (X, Y) -> T
        # Split into feature extractor and head for ordinal support
        if k is not None:
            self.C_features = make_mlp(
                [d_x + d_y, *hidden_dims],
                activation=activation,
                dropout=dropout,
                norm_layer=norm_layer,
                residual=residual,
            )
            c_in_dim = hidden_dims[-1] if hidden_dims else d_x + d_y
            if ordinal:
                self.C_head = OrdinalHead(c_in_dim, k, method=ordinal_method)
            else:
                self.C_head = nn.Linear(c_in_dim, k)
        else:
            # For continuous treatment, keep the original structure
            self.C_features = make_mlp(
                [d_x + d_y, *hidden_dims],
                activation=activation,
                dropout=dropout,
                norm_layer=norm_layer,
                residual=residual,
            )
            c_in_dim = hidden_dims[-1] if hidden_dims else d_x + d_y
            self.C_head = nn.Linear(c_in_dim, 1)

    # ------------------------------------------------------------------
    def forward(self, X: torch.Tensor, T: torch.Tensor):
        """Predict outcome ``Y`` from covariates ``X`` and treatment ``T``."""

        if self.k is not None:
            T_1h = self.t_embedding(T.to(torch.long))
        else:
            T_1h = T.unsqueeze(-1).float()
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
        T_obs : (B,) int in ``[0, K-1]`` or float if labelled, ``-1`` if unknown
        """
        labelled = T_obs >= 0
        unlabelled = ~labelled

        # === STEP 1 : make *some* treatment for every row =============
        XY = torch.cat([X, Y], -1)
        c_features = self.C_features(XY)
        logits_T = self.C_head(c_features)
        if self.k is not None:
            # For ordinal heads, we need class probabilities for prediction
            if self.ordinal and self.ordinal_method in ["coral", "cumulative"]:
                probs_T = self.C_head.predict_proba(c_features)
                T_pred = probs_T.argmax(-1)
            else:
                T_pred = logits_T.argmax(-1)
            T_use = torch.where(labelled, T_obs.to(torch.long), T_pred)
            T_1h = self.t_embedding(T_use)
        else:
            T_pred = logits_T.squeeze(-1)
            T_use = torch.where(labelled, T_obs.float(), T_pred)
            T_1h = T_use.unsqueeze(-1)
        XT = torch.cat([X, T_1h], -1)
        TY = torch.cat([T_1h, Y], -1)

        # === STEP 2 : forward + backward generators ===================
        if self.lowrank_head:
            h_y = self.G_Y(XT)
            mu, Fmat, sigma2 = self.G_Y_head(h_y)
            Y_hat = mu
        else:
            Y_hat = self.G_Y(XT)
            mu = Fmat = sigma2 = None
        X_hat = self.G_X(TY)

        # cycles
        Y_hat_detached = Y_hat.detach()
        X_hat_detached = X_hat.detach()
        TY_hat = torch.cat([T_1h, Y_hat_detached], -1)
        X_cyc = self.G_X(TY_hat)
        Xhat_T = torch.cat([X_hat_detached, T_1h], -1)
        if self.lowrank_head:
            h_cyc = self.G_Y(Xhat_T)
            mu_cyc, F_cyc, sigma2_cyc = self.G_Y_head(h_cyc)
            Y_cyc = mu_cyc
        else:
            Y_cyc = self.G_Y(Xhat_T)

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
        if self.k is not None:
            if labelled.any():
                # Use ordinal losses if ordinal mode is enabled
                if self.ordinal:
                    if self.ordinal_method == "coral":
                        L_sup_T = coral_loss(logits_T[labelled], T_obs[labelled].to(torch.long), self.k)
                    elif self.ordinal_method == "cumulative":
                        L_sup_T = cumulative_link_loss(logits_T[labelled], T_obs[labelled].to(torch.long), self.k)
                    else:
                        L_sup_T = cross_entropy_loss(logits_T[labelled], T_obs[labelled].to(torch.long))
                else:
                    L_sup_T = cross_entropy_loss(logits_T[labelled], T_obs[labelled].to(torch.long))
            else:
                L_sup_T = 0.0
        else:
            L_sup_T = (
                mse(logits_T[labelled].squeeze(-1), T_obs[labelled].float())
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
        if unlabelled.any() and self.k is not None:
            # Get class probabilities for entropy computation
            if self.ordinal and self.ordinal_method in ["coral", "cumulative"]:
                c_features_ulb = self.C_features(XY[unlabelled])
                P_ulb = self.C_head.predict_proba(c_features_ulb)
            else:
                P_ulb = logits_T[unlabelled].softmax(-1)
            L_ent = -(P_ulb * (P_ulb + 1e-8).log()).sum(-1).mean()
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

        c_features = self.C_features(torch.cat([X, Y], dim=-1))
        if self.k is None:
            return self.C_head(c_features).squeeze(-1)
        # Use ordinal predict_proba if ordinal mode is enabled
        if self.ordinal and self.ordinal_method in ["coral", "cumulative"]:
            return self.C_head.predict_proba(c_features)
        return self.C_head(c_features).softmax(dim=-1)


__all__ = ["CycleDual"]

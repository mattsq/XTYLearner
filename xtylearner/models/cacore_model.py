import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import make_mlp
from .registry import register_model


@register_model("cacore")
class CaCoRE(nn.Module):
    """Contrastive Causal Representation Encoder.

    Maps covariates ``x`` to a latent representation ``h(x)`` shared between an
    outcome head and a propensity head. A contrastive InfoNCE objective links
    ``h(x)`` with a joint embedding of ``(y, t)`` to maximise mutual
    information.
    """

    def __init__(
        self,
        d_x: int,
        d_y: int,
        k: int,
        hidden: int = 128,
        rep_dim: int = 64,
        cpc_weight: float = 1.0,
    ) -> None:
        """Initialize the CaCoRE networks.

        Parameters
        ----------
        d_x: int
            Number of covariate features.
        d_y: int
            Dimensionality of the outcome.
        k: int
            Number of treatment classes.
        hidden: int, optional
            Width of the hidden layers. Defaults to ``128``.
        rep_dim: int, optional
            Size of the latent representation ``h(x)``. Defaults to ``64``.
        cpc_weight: float, optional
            Weight of the InfoNCE regulariser. Defaults to ``1.0``.
        """

        super().__init__()
        self.k = k
        self.d_y = d_y
        self.cpc_weight = cpc_weight

        # Encoder mapping x -> h
        self.encoder = make_mlp([d_x, hidden, rep_dim], activation=nn.ReLU)

        # Outcome prediction head: (h, t) -> y
        self.outcome_head = make_mlp([
            rep_dim + k,
            hidden,
            d_y,
        ], activation=nn.ReLU)

        # Propensity head: (h, y) -> t logits
        self.propensity_head = make_mlp([
            rep_dim + d_y,
            hidden,
            k,
        ], activation=nn.ReLU)

        # Joint embedding for contrastive loss
        self.joint_embed = make_mlp([
            d_y + k,
            hidden,
            rep_dim,
        ], activation=nn.ReLU)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return outcome prediction ``y`` or representation ``h(x)``.

        Parameters
        ----------
        x : torch.Tensor
            Covariate batch of shape ``(n, d_x)``.
        t : torch.Tensor | None, optional
            Treatment indices in ``[0, k-1]``.  If ``None`` (default) the
            latent representation ``h(x)`` is returned instead of an outcome
            prediction.
        """

        h = self.encoder(x)
        if t is None:
            return h
        t_oh = F.one_hot(t, num_classes=self.k).float()
        h_t = torch.cat([h, t_oh], dim=-1)
        return self.outcome_head(h_t)

    # ------------------------------------------------------------------
    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the training loss.

        Parameters
        ----------
        x : torch.Tensor
            Covariates of shape ``(batch, d_x)``.
        y : torch.Tensor
            Outcomes with shape ``(batch, d_y)``.
        t_obs : torch.Tensor
            Observed treatments in ``[0, k-1]`` or ``-1`` when missing.

        Returns
        -------
        torch.Tensor
            Scalar loss combining outcome, propensity and contrastive terms.
        """
        y = y.float()
        h = self.encoder(x)

        # Outcome loss only on rows with observed treatment
        labelled = t_obs >= 0
        t_onehot = F.one_hot(t_obs.clamp_min(0), num_classes=self.k).float()
        h_t = torch.cat([h, t_onehot], dim=-1)
        y_hat = self.outcome_head(h_t)
        loss_y = torch.tensor(0.0, device=x.device)
        if labelled.any():
            loss_y = F.mse_loss(y_hat[labelled], y[labelled])

        # Treatment prediction loss
        h_y = torch.cat([h, y], dim=-1)
        logits_t = self.propensity_head(h_y)
        loss_t = torch.tensor(0.0, device=x.device)
        if labelled.any():
            loss_t = F.cross_entropy(logits_t[labelled], t_obs[labelled])

        # Contrastive InfoNCE over (y,t)
        z = self.joint_embed(torch.cat([y, t_onehot], dim=-1))
        z = F.normalize(z, dim=-1)
        h_norm = F.normalize(h, dim=-1)
        sim = h_norm @ z.T
        labels = torch.arange(x.size(0), device=x.device)
        loss_cpc = F.cross_entropy(sim, labels)

        return loss_y + loss_t + self.cpc_weight * loss_cpc

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_outcome(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict outcome ``y`` for covariates ``x`` under treatment ``t``."""

        h = self.encoder(x)
        t_oh = F.one_hot(t, num_classes=self.k).float()
        h_t = torch.cat([h, t_oh], dim=-1)
        return self.outcome_head(h_t)

    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return ``p(t|x,y)`` from the propensity head."""

        h = self.encoder(x)
        h_y = torch.cat([h, y.float()], dim=-1)
        logits = self.propensity_head(h_y)
        return F.softmax(logits, dim=-1)


__all__ = ["CaCoRE"]

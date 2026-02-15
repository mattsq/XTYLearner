"""Deterministic Gaussian Autoencoder with Wristband loss.

Uses the Wristband Gaussian Loss to push deterministic latent representations
toward N(0, I), enabling composable and independent latent factors for
counterfactual reasoning.

The model encodes covariates X through Euclidean attention and an invertible
flow to produce latent representations that are close to standard normal.
Separate latent blocks for different factor groups (e.g. treatment vs.
covariates) can be recombined or resampled for counterfactual queries.

Reference
---------
Adapted from ``DeterministicGAE.py`` in https://github.com/mvparakhin/ml-tidbits
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model
from .embed_layers import C_EmbedAttentionModule, C_ACN, C_InvertibleFlow
from ..losses import C_WristbandGaussianLoss


@register_model("deterministic_gae")
class DeterministicGAE(nn.Module):
    """Deterministic Gaussian Autoencoder for causal factor learning.

    Combines Euclidean attention encoding, invertible normalizing flows, and
    the Wristband Gaussian Loss to learn a deterministic mapping from
    covariates to a latent space close to N(0, I).

    The model learns per-treatment outcome predictions from the latent space,
    supporting counterfactual queries by resampling latent blocks.

    Parameters
    ----------
    d_x : int
        Covariate dimensionality.
    d_y : int
        Outcome dimensionality.
    k : int | None
        Number of discrete treatment arms.  When ``None`` the treatment is
        assumed continuous and a single outcome head is used.
    embed_dim : int
        Latent embedding dimensionality.
    n_heads : int
        Number of attention heads in encoder/decoder.
    n_basis : int
        Number of learnable key/value prototypes per head.
    internal_dim : int
        Per-head value dimensionality in attention.
    flow_layers : int
        Number of coupling layers in the invertible flow.
    flow_hidden : int
        Hidden width in each flow conditioner.
    flow_blocks : int
        Residual blocks per flow conditioner.
    lambda_rec : float
        Weight for the reconstruction loss.
    lambda_wb : float
        Weight for the Wristband Gaussian Loss on latents.
    calibration_reps : int
        Monte-Carlo repetitions for Wristband calibration.
    batch_size_hint : int
        Expected batch size, used for Wristband calibration shape.
    """

    trainer_type = "supervised"

    def __init__(
        self,
        d_x: int,
        d_y: int = 1,
        k: int | None = 2,
        embed_dim: int = 8,
        n_heads: int = 64,
        n_basis: int = 256,
        internal_dim: int = 128,
        flow_layers: int = 4,
        flow_hidden: int = 32,
        flow_blocks: int = 2,
        lambda_rec: float = 1.0,
        lambda_wb: float = 0.1,
        calibration_reps: int = 512,
        batch_size_hint: int = 256,
    ) -> None:
        super().__init__()
        self.d_x = d_x
        self.d_y = d_y
        self.k = k
        self.embed_dim = embed_dim
        self.lambda_rec = lambda_rec
        self.lambda_wb = lambda_wb

        # Encoder: x -> embedding via Euclidean attention
        self.encoder = C_EmbedAttentionModule(
            d_x,
            internal_dim,
            embed_dim,
            n_basis,
            n_heads,
            q_transform=nn.Linear(d_x, d_x),
            head_combine=C_ACN(n_heads * internal_dim, embed_dim, internal_dim, 2),
        )

        # Decoder: embedding -> reconstruction of x
        self.decoder = C_EmbedAttentionModule(
            embed_dim,
            internal_dim,
            d_x,
            n_basis,
            n_heads,
            head_combine=C_ACN(n_heads * internal_dim, d_x, internal_dim, 2),
        )

        # Invertible flow: warp encoding toward N(0,I)
        self.flow = C_InvertibleFlow(
            embed_dim,
            n_layers=flow_layers,
            hidden_dim=flow_hidden,
            n_blocks=flow_blocks,
            s_max=2.0,
            permute_mode="per_pair",
        )

        # Outcome heads: predict y from latent
        if k is not None and k > 0:
            self.outcome_head = nn.Linear(embed_dim, k * d_y)
        else:
            # Continuous treatment: single head that takes [z, t]
            self.outcome_head = nn.Linear(embed_dim + 1, d_y)

        # Treatment classifier: p(t | x, y) from latent + outcome
        if k is not None and k > 0:
            self.cls_t = nn.Linear(embed_dim + d_y, k)

        # Wristband loss (calibrated for expected batch/embed shape)
        self.wristband = C_WristbandGaussianLoss(
            calibration_shape=(batch_size_hint, embed_dim),
            calibration_reps=calibration_reps,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode covariates to latent space through encoder + flow.

        Parameters
        ----------
        x : Tensor
            Covariates of shape ``(batch, d_x)``.

        Returns
        -------
        Tensor
            Latent codes of shape ``(batch, embed_dim)``.
        """
        return self.flow(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent codes back to covariate space.

        Parameters
        ----------
        z : Tensor
            Latent codes of shape ``(batch, embed_dim)``.

        Returns
        -------
        Tensor
            Reconstructed covariates of shape ``(batch, d_x)``.
        """
        return self.decoder(self.flow.inverse(z))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict outcomes for covariates ``x`` under treatment ``t``.

        Parameters
        ----------
        x : Tensor
            Covariates of shape ``(batch, d_x)``.
        t : Tensor
            Treatment indicators.

        Returns
        -------
        Tensor
            Predicted outcomes of shape ``(batch, d_y)``.
        """
        z = self.encode(x)

        if self.k is not None and self.k > 0:
            mu = self.outcome_head(z).view(-1, self.k, self.d_y)
            t_idx = t.long().view(-1, 1, 1).expand(-1, 1, self.d_y)
            return mu.gather(1, t_idx).squeeze(1)
        else:
            t_inp = t.float().view(-1, 1) if t.dim() == 1 else t.float()
            return self.outcome_head(torch.cat([z, t_inp], dim=-1))

    def loss(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> torch.Tensor:
        """Compute the combined training loss.

        The loss is a weighted combination of:
        - Outcome prediction loss (MSE for labelled samples)
        - Covariate reconstruction loss (MSE)
        - Wristband Gaussian loss on latent codes

        Parameters
        ----------
        x : Tensor
            Covariates of shape ``(batch, d_x)``.
        y : Tensor
            Outcomes of shape ``(batch, d_y)``.
        t_obs : Tensor
            Observed treatment labels.  ``-1`` marks unlabelled samples.

        Returns
        -------
        Tensor
            Scalar training loss.
        """
        z = self.encode(x)
        x_hat = self.decode(z)

        # Reconstruction loss
        rec_loss = (x_hat - x).square().mean()

        # Wristband loss on latents
        wb = self.wristband(z)

        # Outcome prediction loss (labelled samples only)
        labelled = t_obs >= 0
        outcome_loss = torch.tensor(0.0, device=x.device)
        if labelled.any():
            t_lab = t_obs[labelled]
            y_lab = y[labelled]

            if self.k is not None and self.k > 0:
                mu = self.outcome_head(z[labelled]).view(-1, self.k, self.d_y)
                t_idx = t_lab.long().view(-1, 1, 1).expand(-1, 1, self.d_y)
                y_pred = mu.gather(1, t_idx).squeeze(1)
            else:
                t_inp = t_lab.float().view(-1, 1)
                y_pred = self.outcome_head(
                    torch.cat([z[labelled], t_inp], dim=-1)
                )

            outcome_loss = F.mse_loss(y_pred.squeeze(-1), y_lab.squeeze(-1))

        total = outcome_loss + self.lambda_rec * rec_loss + self.lambda_wb * wb.total
        return total

    @torch.no_grad()
    def predict_outcome(
        self, x: torch.Tensor, t: int | torch.Tensor
    ) -> torch.Tensor:
        """Predict outcome for covariates ``x`` under treatment ``t``.

        Parameters
        ----------
        x : Tensor
            Covariates of shape ``(batch, d_x)``.
        t : int | Tensor
            Treatment index (scalar) or per-sample tensor.

        Returns
        -------
        Tensor
            Predicted outcomes.
        """
        z = self.encode(x)

        if self.k is not None and self.k > 0:
            mu = self.outcome_head(z).view(-1, self.k, self.d_y)
            if isinstance(t, int):
                return mu[:, t, :].squeeze(-1)
            if isinstance(t, torch.Tensor) and t.dim() == 0:
                return mu[:, int(t.item()), :].squeeze(-1)
            t_idx = t.long().view(-1, 1, 1).expand(-1, 1, self.d_y)
            return mu.gather(1, t_idx).squeeze(1)
        else:
            if isinstance(t, (int, float)):
                t_inp = torch.full((x.size(0), 1), float(t), device=x.device)
            elif isinstance(t, torch.Tensor) and t.dim() == 0:
                t_inp = torch.full(
                    (x.size(0), 1), t.float().item(), device=x.device
                )
            else:
                t_inp = t.float().view(-1, 1)
            return self.outcome_head(torch.cat([z, t_inp], dim=-1))

    @torch.no_grad()
    def predict_treatment_proba(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return treatment probabilities ``p(t | x, y)``.

        Parameters
        ----------
        x : Tensor
            Covariates of shape ``(batch, d_x)``.
        y : Tensor | None
            Outcomes of shape ``(batch, d_y)``.  Ignored when ``k`` is None.

        Returns
        -------
        Tensor
            Treatment probability matrix of shape ``(batch, k)``.
        """
        if self.k is None or self.k <= 0:
            return torch.ones(x.size(0), 1, device=x.device)
        z = self.encode(x)
        if y is None:
            y = torch.zeros(x.size(0), self.d_y, device=x.device)
        y_flat = y.view(x.size(0), -1)
        logits = self.cls_t(torch.cat([z, y_flat], dim=-1))
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_counterfactual(
        self,
        x: torch.Tensor,
        t: int | torch.Tensor,
        n_samples: int = 100,
    ) -> torch.Tensor:
        """Estimate counterfactual outcome distribution by resampling latents.

        Encodes ``x`` to get the deterministic latent, then replaces a portion
        of latent dimensions with fresh N(0, I) draws to simulate variation in
        uncontrolled factors.  The outcome predictions across samples give an
        empirical distribution.

        Parameters
        ----------
        x : Tensor
            Covariates of shape ``(batch, d_x)``.
        t : int | Tensor
            Treatment to condition on.
        n_samples : int
            Number of Monte-Carlo samples.

        Returns
        -------
        Tensor
            Predicted outcomes of shape ``(n_samples, batch, d_y)``.
        """
        z = self.encode(x)
        batch = z.shape[0]
        results = []

        for _ in range(n_samples):
            # Resample half the latent dims (simulating uncontrolled factors)
            noise = torch.randn_like(z)
            half = self.embed_dim // 2
            z_cf = torch.cat([z[:, :half], noise[:, half:]], dim=-1)

            if self.k is not None and self.k > 0:
                mu = self.outcome_head(z_cf).view(-1, self.k, self.d_y)
                if isinstance(t, int):
                    y_pred = mu[:, t, :].squeeze(-1)
                elif isinstance(t, torch.Tensor) and t.dim() == 0:
                    y_pred = mu[:, int(t.item()), :].squeeze(-1)
                else:
                    t_idx = t.long().view(-1, 1, 1).expand(-1, 1, self.d_y)
                    y_pred = mu.gather(1, t_idx).squeeze(1)
            else:
                if isinstance(t, int):
                    t_inp = torch.full((batch, 1), float(t), device=z.device)
                elif isinstance(t, torch.Tensor) and t.dim() == 0:
                    t_inp = torch.full(
                        (batch, 1), t.float().item(), device=z.device
                    )
                else:
                    t_inp = t.float().view(-1, 1)
                y_pred = self.outcome_head(torch.cat([z_cf, t_inp], dim=-1))

            results.append(y_pred)

        return torch.stack(results, dim=0)


__all__ = ["DeterministicGAE"]

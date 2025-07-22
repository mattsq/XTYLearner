import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model
from .utils import reparameterise, kl_normal


@register_model("scgm")
class SCGM(nn.Module):
    """Semi-Supervised Causal Generative Model."""

    def __init__(self, d_x: int, d_y: int = 1, k: int = 2, z_dim: int = 32, h: int = 128) -> None:
        super().__init__()
        self.k = k
        self.d_y = d_y
        # inference networks
        enc_z_in = d_x + k + 1 + 2 * d_y
        self.enc_z = nn.Sequential(
            nn.Linear(enc_z_in, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )
        self.z_mu = nn.Linear(h, z_dim)
        self.z_logvar = nn.Linear(h, z_dim)

        enc_t_in = d_x + 2 * d_y
        self.enc_t = nn.Sequential(
            nn.Linear(enc_t_in, h),
            nn.ReLU(),
            nn.Linear(h, k),
        )

        # generative networks
        self.dec_x = nn.Sequential(
            nn.Linear(z_dim, h),
            nn.ReLU(),
            nn.Linear(h, d_x),
        )
        self.dec_t = nn.Sequential(
            nn.Linear(z_dim + d_x, h),
            nn.ReLU(),
            nn.Linear(h, k),
        )
        self.dec_y = nn.Sequential(
            nn.Linear(z_dim + d_x + k, h),
            nn.ReLU(),
            nn.Linear(h, d_y),
        )

    # --------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t_onehot: torch.Tensor,
        y: torch.Tensor,
        m_t: torch.Tensor,
        m_y: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        m_t = m_t if m_t.dim() == 2 else m_t.unsqueeze(-1)
        m_y = m_y if m_y.dim() == 2 else m_y.unsqueeze(-1)
        enc_in = torch.cat(
            [x, y * m_y, t_onehot * m_t, m_t, m_y], dim=-1
        )
        h_z = self.enc_z(enc_in)
        z_mu = self.z_mu(h_z)
        z_logvar = self.z_logvar(h_z)
        z = reparameterise(z_mu, z_logvar)

        x_hat = self.dec_x(z)
        t_logits = self.dec_t(torch.cat([z, x], dim=-1))
        y_hat = self.dec_y(torch.cat([z, x, t_onehot], dim=-1))

        t_post = self.enc_t(torch.cat([x, y * m_y, m_y], dim=-1))

        return {
            "z": z,
            "x_hat": x_hat,
            "t_logits": t_logits,
            "y_hat": y_hat,
            "t_post": t_post,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
        }

    # --------------------------------------------------------------
    def loss(self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        if t_obs.dim() == 2 and t_obs.size(1) == 1:
            t_obs = t_obs.squeeze(1)
        mask_t = t_obs >= 0
        t_clean = t_obs.clone().long()
        t_clean[~mask_t] = 0
        t_onehot = F.one_hot(t_clean, self.k).float()

        mask_y = ~torch.isnan(y)
        y_filled = torch.nan_to_num(y, nan=0.0)

        out = self.forward(
            x,
            t_onehot,
            y_filled,
            mask_t.float().unsqueeze(-1),
            mask_y.float(),
        )

        log_px = -F.mse_loss(out["x_hat"], x, reduction="none").sum(-1)
        log_pt = -F.cross_entropy(out["t_logits"], t_clean, reduction="none")
        log_py = -F.mse_loss(out["y_hat"], y_filled, reduction="none").sum(-1)

        zeros = torch.zeros_like(out["z_mu"])
        kl_z = kl_normal(out["z_mu"], out["z_logvar"], zeros, zeros)
        kl_t = F.kl_div(
            out["t_logits"].log_softmax(-1),
            out["t_post"].softmax(-1),
            reduction="none",
        ).sum(-1)

        elbo = (
            log_px
            + mask_t.float() * log_pt
            + mask_y.float() * log_py
            - kl_z
            - (1.0 - mask_t.float()) * kl_t
        )
        loss = -elbo.mean()
        return {"loss": loss}

    # --------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mask_y = ~torch.isnan(y)
        y_f = torch.nan_to_num(y, nan=0.0)
        logits = self.enc_t(torch.cat([x, y_f, mask_y.float()], dim=-1))
        return logits.softmax(-1)

    @torch.no_grad()
    def predict_outcome(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1 or (t.dim() == 2 and t.size(1) == 1):
            t_onehot = F.one_hot(t.long().view(-1), self.k).float()
        else:
            t_onehot = t.float()
        z = torch.zeros(x.size(0), self.z_mu.out_features, device=x.device)
        return self.dec_y(torch.cat([z, x, t_onehot], dim=-1))

    def forward_outcome(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.predict_outcome(x, t)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.predict_outcome(x, t)


__all__ = ["SCGM"]

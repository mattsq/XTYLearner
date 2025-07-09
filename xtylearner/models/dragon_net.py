import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import make_mlp
from .registry import register_model


@register_model("dragon_net")
class DragonNet(nn.Module):
    """Shared representation with outcome and propensity heads.

    Includes a reconstruction head ``p(t|x,y)`` and a targeted regularisation
    term for doubly robust effect estimates.
    """

    def __init__(self, d_x: int, d_y: int = 1, k: int = 2, h: int = 200, *, lmbda=(1.0, 1.0, 1.0, 1.0)) -> None:
        super().__init__()
        self.k = k
        self.encoder = make_mlp([d_x, h, h, h], activation=nn.ReLU)
        self.outcome_head = nn.Linear(h, k * d_y)
        self.propensity_head = nn.Linear(h, k)
        self.recon_head = nn.Linear(h + d_y, k)
        self.lmbda = lmbda
        self.d_y = d_y

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        phi = self.encoder(x)
        out = self.outcome_head(phi).view(-1, self.k, self.d_y)
        return out.gather(1, t.view(-1, 1, 1).expand(-1, 1, self.d_y)).squeeze(1)

    # ------------------------------------------------------------------
    def loss(self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor) -> torch.Tensor:
        phi = self.encoder(x)
        mu_hat = self.outcome_head(phi).view(-1, self.k, self.d_y)
        pi_hat = self.propensity_head(phi)
        rho_hat = self.recon_head(torch.cat([phi, y], dim=1))

        labelled = t_obs >= 0
        t_lab = t_obs[labelled]
        y_lab = y[labelled]

        mse_y = torch.tensor(0.0, device=x.device)
        ce_pi = torch.tensor(0.0, device=x.device)
        ce_rho = torch.tensor(0.0, device=x.device)
        if labelled.any():
            mse_y = F.mse_loss(
                mu_hat[labelled, t_lab, :].squeeze(-1), y_lab.squeeze(-1)
            )
            ce_pi = F.cross_entropy(pi_hat[labelled], t_lab)
            ce_rho = F.cross_entropy(rho_hat[labelled], t_lab)

        unlabelled = ~labelled
        kl = torch.tensor(0.0, device=x.device)
        if unlabelled.any():
            kl = F.kl_div(
                F.log_softmax(pi_hat[unlabelled], dim=-1),
                F.softmax(rho_hat[unlabelled].detach(), dim=-1),
                reduction="batchmean",
            )

        tar_reg = torch.tensor(0.0, device=x.device)
        if labelled.any():
            t_onehot = F.one_hot(t_lab, self.k).float()
            mu_lab = mu_hat[labelled]
            tau_dr = (
                (t_onehot / pi_hat[labelled]).unsqueeze(-1)
                * (y_lab.unsqueeze(1) - mu_lab)
            ).mean(0)
            tar_reg = tau_dr.pow(2).sum()

        l1, l2, l3, l4 = self.lmbda
        return mse_y + l1 * ce_pi + l2 * ce_rho + l3 * kl + l4 * tar_reg

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_outcome(self, x: torch.Tensor, t: int) -> torch.Tensor:
        phi = self.encoder(x)
        mu = self.outcome_head(phi).view(-1, self.k, self.d_y)
        return mu[:, t, :].squeeze(-1)

    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        phi = self.encoder(x)
        if y is None:
            return F.softmax(self.propensity_head(phi), dim=-1)
        phi_y = torch.cat([phi, y], dim=1)
        return F.softmax(self.recon_head(phi_y), dim=-1)


__all__ = ["DragonNet"]

from __future__ import annotations

import torch
import torch.nn.functional as F


# --- tabular augmentations -------------------------------------------
def weak_aug(x: torch.Tensor) -> torch.Tensor:
    """Light Gaussian noise."""
    return x + 0.01 * torch.randn_like(x)


def strong_aug(x: torch.Tensor) -> torch.Tensor:
    """MixUp and feature dropout."""
    lam = torch.distributions.Beta(1.0, 1.0).sample().item()
    idx = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[idx]
    mask = torch.rand_like(x).bernoulli_(0.15)
    noise = torch.randn_like(x) * x.std(0, keepdim=True)
    return torch.where(mask.bool(), noise, x_mix)


# --- FixMatch loss helper --------------------------------------------
def fixmatch_unsup_loss(model, x_u: torch.Tensor, tau: float = 0.95) -> torch.Tensor:
    """Unsupervised FixMatch loss for a batch of inputs."""
    with torch.no_grad():
        p_w = F.softmax(model(weak_aug(x_u)), dim=1)
        max_p, y_star = p_w.max(1)
        mask = max_p.ge(tau).float()

    if mask.sum() == 0:
        return torch.tensor(0.0, device=x_u.device)

    p_s = model(strong_aug(x_u))
    loss = F.cross_entropy(p_s, y_star, reduction="none")
    return (loss * mask).mean()


# --- training loop skeleton ------------------------------------------
def train_fixmatch(
    model: torch.nn.Module,
    loader_lab,
    loader_unlab,
    *,
    epochs: int = 200,
    mu: int = 7,
    tau: float = 0.95,
    lambda_u: float = 1.0,
    lr: float = 3e-4,
    device: str | None = None,
) -> None:
    """Train ``model`` with FixMatch."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    opt = torch.optim.AdamW(model.parameters(), lr)
    it_unlab = iter(loader_unlab)

    for _ in range(epochs):
        for x_l, y_l in loader_lab:
            try:
                x_u = next(it_unlab)
            except StopIteration:
                it_unlab = iter(loader_unlab)
                x_u = next(it_unlab)

            x_l, y_l = x_l.to(device), y_l.to(device)
            x_u = x_u[0].to(device)

            logits_l = model(weak_aug(x_l))
            L_sup = F.cross_entropy(logits_l, y_l)

            L_unsup = fixmatch_unsup_loss(model, x_u, tau)
            loss = L_sup + lambda_u * L_unsup

            opt.zero_grad()
            loss.backward()
            opt.step()

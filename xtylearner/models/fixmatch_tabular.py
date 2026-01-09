from __future__ import annotations

import torch
import torch.nn.functional as F


# --- soft cross-entropy helper ---------------------------------------
def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss with soft targets.

    Parameters
    ----------
    logits:
        Model predictions (before softmax).
    targets:
        Soft target probabilities (already probabilities, not logits).

    Returns
    -------
    torch.Tensor
        Cross-entropy loss for each sample.
    """
    log_probs = F.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1)


# --- tabular augmentations -------------------------------------------
def weak_aug(x: torch.Tensor) -> torch.Tensor:
    """Apply light Gaussian noise to ``x``.

    Parameters
    ----------
    x:
        Input features.

    Returns
    -------
    torch.Tensor
        Noisy tensor with the same shape as ``x``.
    """
    return x + 0.01 * torch.randn_like(x)


def strong_aug(x: torch.Tensor) -> tuple[torch.Tensor, float, torch.Tensor]:
    """Apply MixUp and feature dropout augmentations.

    Parameters
    ----------
    x:
        Input features.

    Returns
    -------
    tuple[torch.Tensor, float, torch.Tensor]
        A tuple containing:
        - Augmented tensor of the same shape as ``x``
        - Lambda value used for MixUp
        - Permutation indices used for MixUp
    """
    lam = torch.distributions.Beta(1.0, 1.0).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    mask = torch.rand_like(x).bernoulli_(0.15)
    noise = torch.randn_like(x) * x.std(0, keepdim=True)
    x_aug = torch.where(mask.bool(), noise, x_mix)
    return x_aug, lam, idx


# --- FixMatch loss helper --------------------------------------------
def fixmatch_unsup_loss(model, x_u: torch.Tensor, tau: float = 0.95) -> torch.Tensor:
    """Unsupervised FixMatch loss for unlabelled data with MixMatch-style label mixing.

    Parameters
    ----------
    model:
        Classification network.
    x_u:
        Unlabelled input batch.
    tau:
        Confidence threshold for pseudo labels.

    Returns
    -------
    torch.Tensor
        Scalar unsupervised loss.
    """
    with torch.no_grad():
        p_w = F.softmax(model(weak_aug(x_u)), dim=1)
        max_p, _ = p_w.max(1)
        mask = max_p.ge(tau).float()

    if mask.sum() == 0:
        return torch.tensor(0.0, device=x_u.device)

    # Apply strong augmentation and get mixing parameters
    x_strong, lam, idx = strong_aug(x_u)

    # Mix the pseudo-label probabilities (MixMatch-style)
    with torch.no_grad():
        targets = lam * p_w + (1 - lam) * p_w[idx]

    # Compute predictions on strongly augmented data
    p_s = model(x_strong)

    # Use soft cross-entropy since targets are now probability distributions
    loss = soft_cross_entropy(p_s, targets)
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
    """Train ``model`` using the FixMatch algorithm.

    Parameters
    ----------
    model:
        Neural network classifier.
    loader_lab:
        Data loader of labelled batches ``(x_l, y_l)``.
    loader_unlab:
        Data loader of unlabelled examples ``(x_u,)``.
    epochs:
        Number of training epochs.
    mu:
        Ratio of unlabelled to labelled batches.
    tau:
        Confidence threshold for pseudo labels.
    lambda_u:
        Weight of the unsupervised loss term.
    lr:
        Learning rate for AdamW optimiser.
    device:
        Optional device identifier.
    """

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

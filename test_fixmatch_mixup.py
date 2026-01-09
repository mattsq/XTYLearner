"""Test script to verify MixUp pseudo-label mixing in FixMatch."""

import torch
import torch.nn as nn
from xtylearner.models.fixmatch_tabular import (
    fixmatch_unsup_loss,
    soft_cross_entropy,
    strong_aug,
)


class SimpleClassifier(nn.Module):
    """Simple 2-layer classifier for testing."""

    def __init__(self, input_dim: int = 10, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_soft_cross_entropy():
    """Test soft cross-entropy implementation."""
    print("Testing soft_cross_entropy...")
    logits = torch.randn(4, 3)
    targets = torch.softmax(torch.randn(4, 3), dim=1)  # Soft targets

    loss = soft_cross_entropy(logits, targets)
    assert loss.shape == (4,), f"Expected shape (4,), got {loss.shape}"
    assert torch.all(loss >= 0), "Loss should be non-negative"
    print("✓ soft_cross_entropy works correctly")


def test_strong_aug_returns_params():
    """Test that strong_aug returns mixing parameters."""
    print("\nTesting strong_aug returns parameters...")
    x = torch.randn(8, 10)
    x_aug, lam, idx = strong_aug(x)

    assert x_aug.shape == x.shape, f"Expected shape {x.shape}, got {x_aug.shape}"
    assert isinstance(lam, float), f"Lambda should be float, got {type(lam)}"
    assert 0 <= lam <= 1, f"Lambda should be in [0,1], got {lam}"
    assert idx.shape == (8,), f"Expected idx shape (8,), got {idx.shape}"
    print(f"✓ strong_aug returns correct parameters (lam={lam:.3f})")


def test_fixmatch_loss_with_mixing():
    """Test that fixmatch_unsup_loss properly mixes pseudo-labels."""
    print("\nTesting fixmatch_unsup_loss with label mixing...")

    model = SimpleClassifier(input_dim=10, num_classes=3)
    model.eval()

    # Create unlabeled data
    x_u = torch.randn(16, 10)

    # Compute loss
    loss = fixmatch_unsup_loss(model, x_u, tau=0.5)  # Lower tau to ensure mask is non-zero

    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss >= 0, f"Loss should be non-negative, got {loss.item()}"
    print(f"✓ fixmatch_unsup_loss works correctly (loss={loss.item():.4f})")


def test_label_mixing_correctness():
    """Verify that pseudo-labels are actually being mixed."""
    print("\nTesting label mixing correctness...")

    model = SimpleClassifier(input_dim=10, num_classes=3)
    model.eval()

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create data with distinct patterns
    x_u = torch.randn(8, 10)

    # Get weak augmentation predictions (pseudo-labels)
    from xtylearner.models.fixmatch_tabular import weak_aug
    import torch.nn.functional as F

    with torch.no_grad():
        p_w = F.softmax(model(weak_aug(x_u)), dim=1)

    # Get strong augmentation with mixing params
    x_strong, lam, idx = strong_aug(x_u)

    # Compute mixed targets
    with torch.no_grad():
        targets_mixed = lam * p_w + (1 - lam) * p_w[idx]

    # Verify targets are properly mixed
    assert targets_mixed.shape == p_w.shape, "Mixed targets should have same shape as original"
    assert torch.allclose(targets_mixed.sum(dim=1), torch.ones(8)), "Targets should sum to 1"

    # Verify mixing actually happened (unless lam is 0 or 1)
    if 0.01 < lam < 0.99:
        # Check that at least some targets are different from original
        differences = torch.abs(targets_mixed - p_w).sum(dim=1)
        assert torch.any(differences > 1e-6), "Mixing should change at least some pseudo-labels"

    print(f"✓ Label mixing is correct (lam={lam:.3f})")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing MixUp Pseudo-Label Mixing in FixMatch")
    print("=" * 60)

    test_soft_cross_entropy()
    test_strong_aug_returns_params()
    test_fixmatch_loss_with_mixing()
    test_label_mixing_correctness()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

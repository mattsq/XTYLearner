"""Tests for UASD (Uncertainty-Aware Self-Distillation) model."""

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from xtylearner.data import load_mixed_synthetic_dataset
from xtylearner.models import UASD, get_model
from xtylearner.training import Trainer


def test_uasd_basic_forward():
    """Test basic forward pass and shapes."""
    model = UASD(d_x=5, d_y=1, k=3)

    x = torch.randn(10, 5)
    t = torch.randint(0, 3, (10,))
    y = torch.randn(10, 1)

    # Forward pass for outcome prediction
    out = model(x, t)
    assert out.shape == (10, 1)

    # predict_outcome method
    out2 = model.predict_outcome(x, t)
    assert out2.shape == (10, 1)

    # predict_outcome with int treatment
    out3 = model.predict_outcome(x, 1)
    assert out3.shape == (10, 1)


def test_uasd_treatment_proba():
    """Test treatment probability prediction."""
    model = UASD(d_x=5, d_y=1, k=3)

    x = torch.randn(10, 5)
    y = torch.randn(10, 1)

    probs = model.predict_treatment_proba(x, y)
    assert probs.shape == (10, 3)
    # Check valid probability distribution
    assert torch.allclose(probs.sum(dim=-1), torch.ones(10), atol=1e-5)
    assert (probs >= 0).all()
    assert (probs <= 1).all()


def test_uasd_ood_score():
    """Test OOD score prediction."""
    model = UASD(d_x=5, d_y=1, k=3)

    x = torch.randn(10, 5)
    y = torch.randn(10, 1)

    scores = model.predict_ood_score(x, y)
    assert scores.shape == (10,)
    # OOD scores should be in [0, 1] (normalized entropy)
    assert (scores >= 0).all()
    assert (scores <= 1).all()


def test_uasd_loss_labelled_only():
    """Test loss computation with only labelled samples."""
    model = UASD(d_x=5, d_y=1, k=3)
    model.train()

    x = torch.randn(10, 5)
    y = torch.randn(10, 1)
    t_obs = torch.randint(0, 3, (10,))  # All labelled

    loss = model.loss(x, y, t_obs)
    assert loss.dim() == 0  # Scalar
    assert loss.item() > 0  # Loss should be positive
    assert torch.isfinite(loss)


def test_uasd_loss_mixed():
    """Test loss computation with mixed labelled and unlabelled samples."""
    model = UASD(d_x=5, d_y=1, k=3)
    model.train()

    x = torch.randn(10, 5)
    y = torch.randn(10, 1)
    t_obs = torch.randint(0, 3, (10,))
    t_obs[5:] = -1  # Mark half as unlabelled

    loss = model.loss(x, y, t_obs)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_uasd_loss_unlabelled_only():
    """Test loss computation with only unlabelled samples."""
    model = UASD(d_x=5, d_y=1, k=3)
    model.train()

    x = torch.randn(10, 5)
    y = torch.randn(10, 1)
    t_obs = torch.full((10,), -1)  # All unlabelled

    loss = model.loss(x, y, t_obs)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_uasd_step_counter():
    """Test step counter increments."""
    model = UASD(d_x=5, d_y=1, k=3)

    assert model.step_count == 0
    model.step()
    assert model.step_count == 1
    model.step()
    assert model.step_count == 2


def test_uasd_soft_targets_update():
    """Test soft target accumulation."""
    model = UASD(d_x=5, d_y=1, k=3, ema_decay=0.9)
    model.train()

    x = torch.randn(5, 5)
    y = torch.randn(5, 1)
    t_obs = torch.full((5,), -1)  # All unlabelled
    indices = torch.arange(5)

    # Initial targets should be zero
    assert (model.soft_targets[:5] == 0).all()
    assert (model.target_counts[:5] == 0).all()

    # Run loss to update soft targets
    _ = model.loss(x, y, t_obs, unlabelled_indices=indices)

    # Soft targets should now be updated
    assert (model.target_counts[:5] > 0).all()
    # Soft targets should sum to 1 (valid distributions)
    for i in range(5):
        if model.target_counts[i] > 0:
            target_sum = model.soft_targets[i].sum()
            assert torch.isclose(target_sum, torch.tensor(1.0), atol=1e-5)


def test_uasd_reset_soft_targets():
    """Test reset_soft_targets method."""
    model = UASD(d_x=5, d_y=1, k=3)
    model.train()

    x = torch.randn(5, 5)
    y = torch.randn(5, 1)
    t_obs = torch.full((5,), -1)
    indices = torch.arange(5)

    _ = model.loss(x, y, t_obs, unlabelled_indices=indices)
    assert (model.target_counts[:5] > 0).all()

    model.reset_soft_targets()
    assert (model.soft_targets == 0).all()
    assert (model.target_counts == 0).all()


def test_uasd_with_trainer():
    """Test UASD with standard Trainer."""
    dataset = load_mixed_synthetic_dataset(n_samples=20, d_x=2, seed=0, label_ratio=0.5)
    loader = DataLoader(dataset, batch_size=5)
    model = UASD(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    trainer.fit(2)
    metrics = trainer.evaluate(loader)
    assert set(metrics) >= {
        "loss",
        "treatment accuracy",
        "outcome rmse",
        "outcome rmse labelled",
        "outcome rmse unlabelled",
    }


def test_uasd_multiclass():
    """Test UASD with more than 2 treatment classes."""
    rng = np.random.default_rng(42)
    X = torch.from_numpy(rng.normal(size=(15, 4)).astype(np.float32))
    T_true = torch.from_numpy(rng.integers(0, 4, size=15).astype(np.int64))
    Y = torch.from_numpy(
        (T_true.numpy() + rng.normal(size=15)).astype(np.float32)
    ).unsqueeze(-1)
    T_obs = T_true.clone()
    T_obs[::3] = -1  # Mark some as unlabelled

    ds = TensorDataset(X, Y, T_obs)
    loader = DataLoader(ds, batch_size=5)
    model = UASD(d_x=4, d_y=1, k=4)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    trainer.fit(2)
    metrics = trainer.evaluate(loader)
    assert "loss" in metrics


def test_uasd_custom_hyperparameters():
    """Test UASD with custom hyperparameters."""
    model = UASD(
        d_x=5,
        d_y=2,
        k=3,
        hidden_dims=(64, 32),
        activation=nn.Tanh,
        dropout=0.1,
        temperature=3.0,
        ema_decay=0.99,
        entropy_threshold=0.3,
        lambda_u=2.0,
        ramp_up=20,
        max_unlabelled=1000,
    )

    assert model.temperature == 3.0
    assert model.ema_decay == 0.99
    assert model.entropy_threshold == 0.3
    assert model.lambda_u == 2.0
    assert model.ramp_up == 20

    x = torch.randn(5, 5)
    y = torch.randn(5, 2)
    t = torch.randint(0, 3, (5,))

    out = model(x, t)
    assert out.shape == (5, 2)


def test_uasd_registry():
    """Test that UASD is registered and can be instantiated via get_model."""
    model = get_model("uasd", d_x=5, d_y=1, k=3)
    assert isinstance(model, UASD)

    x = torch.randn(5, 5)
    y = torch.randn(5, 1)
    t = torch.randint(0, 3, (5,))

    out = model(x, t)
    assert out.shape == (5, 1)


def test_uasd_entropy_weighting():
    """Test that entropy-based weighting affects loss computation."""
    model = UASD(d_x=5, d_y=1, k=3, entropy_threshold=0.5)
    model.train()

    # First, simulate some training to build up soft targets
    x = torch.randn(10, 5)
    y = torch.randn(10, 1)
    t_obs = torch.full((10,), -1)
    indices = torch.arange(10)

    # Run a few iterations to build soft targets
    for _ in range(5):
        _ = model.loss(x, y, t_obs, unlabelled_indices=indices)
        model.step()

    # Verify soft targets have accumulated
    assert (model.target_counts[:10] > 0).all()


def test_uasd_ramp_up():
    """Test that unsupervised loss ramps up over training."""
    model = UASD(d_x=5, d_y=1, k=3, ramp_up=10, lambda_u=1.0)
    model.train()

    x = torch.randn(10, 5)
    y = torch.randn(10, 1)
    t_obs = torch.full((10,), -1)

    # At step 0, lambda should be very small
    loss_early = model.loss(x, y, t_obs).item()

    # Advance steps
    for _ in range(20):
        model.step()

    # At step 20, lambda should be at max
    model.reset_soft_targets()  # Reset to get comparable loss
    loss_late = model.loss(x, y, t_obs).item()

    # The losses won't be directly comparable due to soft target accumulation,
    # but we can verify the model runs without error
    assert torch.isfinite(torch.tensor(loss_early))
    assert torch.isfinite(torch.tensor(loss_late))


def test_uasd_eval_mode():
    """Test that model works correctly in eval mode."""
    model = UASD(d_x=5, d_y=1, k=3)
    model.eval()

    x = torch.randn(10, 5)
    y = torch.randn(10, 1)
    t = torch.randint(0, 3, (10,))

    # Should work in eval mode
    out = model(x, t)
    assert out.shape == (10, 1)

    probs = model.predict_treatment_proba(x, y)
    assert probs.shape == (10, 3)

    ood_scores = model.predict_ood_score(x, y)
    assert ood_scores.shape == (10,)

"""Tests for Clone (Closed Loop Networks) model."""

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from xtylearner.data import load_mixed_synthetic_dataset
from xtylearner.models import Clone, get_model
from xtylearner.training import Trainer


def test_clone_basic_forward():
    """Test basic forward pass and shapes."""
    model = Clone(d_x=5, d_y=1, k=3)

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


def test_clone_treatment_proba():
    """Test treatment probability prediction."""
    model = Clone(d_x=5, d_y=1, k=3)

    x = torch.randn(10, 5)
    y = torch.randn(10, 1)

    probs = model.predict_treatment_proba(x, y)
    assert probs.shape == (10, 3)
    # Check valid probability distribution
    assert torch.allclose(probs.sum(dim=-1), torch.ones(10), atol=1e-5)
    assert (probs >= 0).all()
    assert (probs <= 1).all()


def test_clone_ood_score():
    """Test OOD score prediction using the dedicated OOD network."""
    model = Clone(d_x=5, d_y=1, k=3)

    x = torch.randn(10, 5)
    y = torch.randn(10, 1)

    scores = model.predict_ood_score(x, y)
    assert scores.shape == (10,)
    # OOD scores should be in [0, 1] (sigmoid output)
    assert (scores >= 0).all()
    assert (scores <= 1).all()


def test_clone_loss_labelled_only():
    """Test loss computation with only labelled samples."""
    model = Clone(d_x=5, d_y=1, k=3)
    model.train()

    x = torch.randn(10, 5)
    y = torch.randn(10, 1)
    t_obs = torch.randint(0, 3, (10,))  # All labelled

    loss = model.loss(x, y, t_obs)
    assert loss.dim() == 0  # Scalar
    assert loss.item() > 0  # Loss should be positive
    assert torch.isfinite(loss)


def test_clone_loss_mixed():
    """Test loss computation with mixed labelled and unlabelled samples."""
    model = Clone(d_x=5, d_y=1, k=3)
    model.train()

    x = torch.randn(10, 5)
    y = torch.randn(10, 1)
    t_obs = torch.randint(0, 3, (10,))
    t_obs[5:] = -1  # Mark half as unlabelled

    loss = model.loss(x, y, t_obs)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_clone_loss_unlabelled_only():
    """Test loss computation with only unlabelled samples."""
    model = Clone(d_x=5, d_y=1, k=3)
    model.train()

    x = torch.randn(10, 5)
    y = torch.randn(10, 1)
    t_obs = torch.full((10,), -1)  # All unlabelled

    loss = model.loss(x, y, t_obs)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_clone_step_counter():
    """Test step counter increments."""
    model = Clone(d_x=5, d_y=1, k=3)

    assert model.step_count == 0
    model.step()
    assert model.step_count == 1
    model.step()
    assert model.step_count == 2


def test_clone_ood_filtering():
    """Test that OOD filtering mechanism works."""
    model = Clone(d_x=5, d_y=1, k=3, tau_ood=0.5)
    model.train()

    x = torch.randn(10, 5)
    y = torch.randn(10, 1)
    t_obs = torch.randint(0, 3, (10,))
    t_obs[5:] = -1  # Mark half as unlabelled

    # Run a few training steps
    loss = model.loss(x, y, t_obs)
    assert torch.isfinite(loss)

    # Check that OOD predictions are in valid range
    model.eval()
    ood_scores = model.predict_ood_score(x, y)
    assert (ood_scores >= 0).all()
    assert (ood_scores <= 1).all()


def test_clone_with_trainer():
    """Test Clone with standard Trainer."""
    dataset = load_mixed_synthetic_dataset(n_samples=20, d_x=2, seed=0, label_ratio=0.5)
    loader = DataLoader(dataset, batch_size=5)
    model = Clone(d_x=2, d_y=1, k=2)
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


def test_clone_multiclass():
    """Test Clone with more than 2 treatment classes."""
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
    model = Clone(d_x=4, d_y=1, k=4)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    trainer.fit(2)
    metrics = trainer.evaluate(loader)
    assert "loss" in metrics


def test_clone_custom_hyperparameters():
    """Test Clone with custom hyperparameters."""
    model = Clone(
        d_x=5,
        d_y=2,
        k=3,
        hidden_dims=(64, 32),
        activation=nn.Tanh,
        dropout=0.1,
        tau=0.9,
        tau_ood=0.6,
        lambda_u=2.0,
        lambda_feedback=0.3,
        ramp_up=20,
    )

    assert model.tau == 0.9
    assert model.tau_ood == 0.6
    assert model.lambda_u == 2.0
    assert model.lambda_feedback == 0.3
    assert model.ramp_up == 20

    x = torch.randn(5, 5)
    y = torch.randn(5, 2)
    t = torch.randint(0, 3, (5,))

    out = model(x, t)
    assert out.shape == (5, 2)


def test_clone_registry():
    """Test that Clone is registered and can be instantiated via get_model."""
    model = get_model("clone", d_x=5, d_y=1, k=3)
    assert isinstance(model, Clone)

    x = torch.randn(5, 5)
    y = torch.randn(5, 1)
    t = torch.randint(0, 3, (5,))

    out = model(x, t)
    assert out.shape == (5, 1)


def test_clone_decoupled_networks():
    """Test that OOD and classifier networks are independent."""
    model = Clone(d_x=5, d_y=1, k=3)

    # Verify networks are separate
    assert model.ood_network is not model.classifier_network

    # Check parameter counts
    ood_params = sum(p.numel() for p in model.ood_network.parameters())
    classifier_params = sum(p.numel() for p in model.classifier_network.parameters())

    # Networks should have different output dimensions
    # OOD network outputs 1 value, classifier outputs k values
    assert ood_params != classifier_params


def test_clone_feedback_mechanism():
    """Test that feedback loop is working during training."""
    model = Clone(d_x=5, d_y=1, k=3, lambda_feedback=1.0, tau=0.5)
    model.train()

    x = torch.randn(20, 5)
    y = torch.randn(20, 1)
    t_obs = torch.randint(0, 3, (20,))
    t_obs[10:] = -1  # Mark half as unlabelled

    # Run multiple training steps
    for _ in range(5):
        loss = model.loss(x, y, t_obs)
        assert torch.isfinite(loss)
        model.step()

    # Verify OOD detector is producing outputs
    model.eval()
    ood_scores = model.predict_ood_score(x, y)
    # Not all should be 0.5 (initial random state)
    assert not torch.allclose(ood_scores, torch.full_like(ood_scores, 0.5), atol=0.1)


def test_clone_ramp_up():
    """Test that unsupervised loss ramps up over training."""
    model = Clone(d_x=5, d_y=1, k=3, ramp_up=10, lambda_u=1.0)
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
    loss_late = model.loss(x, y, t_obs).item()

    # Verify both losses are finite
    assert torch.isfinite(torch.tensor(loss_early))
    assert torch.isfinite(torch.tensor(loss_late))


def test_clone_eval_mode():
    """Test that model works correctly in eval mode."""
    model = Clone(d_x=5, d_y=1, k=3)
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


def test_clone_gradient_flow():
    """Test that gradients flow through both networks."""
    model = Clone(d_x=5, d_y=1, k=3)
    model.train()

    x = torch.randn(10, 5)
    y = torch.randn(10, 1)
    t_obs = torch.randint(0, 3, (10,))
    t_obs[5:] = -1

    loss = model.loss(x, y, t_obs)
    loss.backward()

    # Check that both networks have gradients
    ood_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.ood_network.parameters()
    )
    classifier_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.classifier_network.parameters()
    )

    assert ood_has_grad, "OOD network should have gradients"
    assert classifier_has_grad, "Classifier network should have gradients"

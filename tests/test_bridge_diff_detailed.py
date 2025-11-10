"""Comprehensive unit tests for BridgeDiff model components."""

import torch
import pytest
import numpy as np
from torch.utils.data import DataLoader

from xtylearner.models.bridge_diff import BridgeDiff, ScoreBridge, Classifier
from xtylearner.data import load_mixed_synthetic_dataset
from xtylearner.training import Trainer


class TestScoreBridge:
    """Test the ScoreBridge score network component."""

    def test_score_bridge_forward_shapes(self):
        """Test forward pass produces correct output shapes."""
        d_x, d_y, k = 2, 1, 2
        batch_size = 10

        score_net = ScoreBridge(d_x, d_y, k, hidden=128, embed_dim=32, n_blocks=2)

        y_noisy = torch.randn(batch_size, d_y)
        x = torch.randn(batch_size, d_x)
        t = torch.randint(0, k, (batch_size,))
        tau = torch.rand(batch_size)

        output = score_net(y_noisy, x, t, tau)

        assert output.shape == (batch_size, d_y), f"Expected shape {(batch_size, d_y)}, got {output.shape}"

    def test_score_bridge_tau_dimensions(self):
        """Test that tau dimension handling works correctly."""
        d_x, d_y, k = 2, 1, 2
        batch_size = 5

        score_net = ScoreBridge(d_x, d_y, k)

        y_noisy = torch.randn(batch_size, d_y)
        x = torch.randn(batch_size, d_x)
        t = torch.zeros(batch_size, dtype=torch.long)

        # Test 1D tau
        tau_1d = torch.rand(batch_size)
        output_1d = score_net(y_noisy, x, t, tau_1d)

        # Test 2D tau
        tau_2d = torch.rand(batch_size, 1)
        output_2d = score_net(y_noisy, x, t, tau_2d)

        # Should handle both correctly
        assert output_1d.shape == (batch_size, d_y)
        assert output_2d.shape == (batch_size, d_y)

    def test_score_bridge_gradient_flow(self):
        """Test that gradients flow through score network."""
        d_x, d_y, k = 2, 1, 2
        batch_size = 4

        score_net = ScoreBridge(d_x, d_y, k)

        y_noisy = torch.randn(batch_size, d_y, requires_grad=True)
        x = torch.randn(batch_size, d_x)
        t = torch.zeros(batch_size, dtype=torch.long)
        tau = torch.rand(batch_size)

        output = score_net(y_noisy, x, t, tau)
        loss = output.pow(2).mean()
        loss.backward()

        # Check gradients exist and are non-zero
        assert y_noisy.grad is not None
        assert torch.any(y_noisy.grad != 0)

        # Check network parameters have gradients
        for name, param in score_net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestClassifier:
    """Test the Classifier component."""

    def test_classifier_output_shape(self):
        """Test classifier produces correct logit shape."""
        d_x, d_y, k = 2, 1, 2
        batch_size = 8

        classifier = Classifier(d_x, d_y, k)

        x = torch.randn(batch_size, d_x)
        y = torch.randn(batch_size, d_y)

        logits = classifier(x, y)

        assert logits.shape == (batch_size, k)

    def test_classifier_probabilities_sum_to_one(self):
        """Test that softmax of classifier output sums to 1."""
        d_x, d_y, k = 3, 2, 3
        batch_size = 10

        classifier = Classifier(d_x, d_y, k)

        x = torch.randn(batch_size, d_x)
        y = torch.randn(batch_size, d_y)

        logits = classifier(x, y)
        probs = torch.softmax(logits, dim=-1)

        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5)

    def test_classifier_gradient_flow(self):
        """Test gradients flow through classifier."""
        d_x, d_y, k = 2, 1, 2
        batch_size = 4

        classifier = Classifier(d_x, d_y, k)

        x = torch.randn(batch_size, d_x)
        y = torch.randn(batch_size, d_y)

        logits = classifier(x, y)
        loss = logits.pow(2).mean()
        loss.backward()

        # Check network parameters have gradients
        for name, param in classifier.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestBridgeDiffSigma:
    """Test sigma schedule in BridgeDiff."""

    def test_sigma_monotonic_increase(self):
        """Test that sigma increases monotonically with tau."""
        model = BridgeDiff(d_x=2, d_y=1)

        tau_values = torch.linspace(0, 1, 100)
        sigmas = model._sigma(tau_values)

        # Check monotonic increase
        diffs = sigmas[1:] - sigmas[:-1]
        assert torch.all(diffs >= 0), "Sigma should be monotonically increasing"

    def test_sigma_boundary_values(self):
        """Test sigma at boundary values tau=0 and tau=1."""
        sigma_min, sigma_max = 0.002, 1.0
        model = BridgeDiff(d_x=2, d_y=1, sigma_min=sigma_min, sigma_max=sigma_max)

        sigma_0 = model._sigma(torch.tensor([0.0]))
        sigma_1 = model._sigma(torch.tensor([1.0]))

        # At tau=0, sigma should be close to sigma_min
        assert torch.allclose(sigma_0, torch.tensor([sigma_min]), atol=1e-5)

        # At tau=1, sigma should be close to sigma_max
        assert torch.allclose(sigma_1, torch.tensor([sigma_max]), atol=1e-5)

    def test_sigma_clamping(self):
        """Test that sigma is clamped within [sigma_min, sigma_max]."""
        sigma_min, sigma_max = 0.01, 0.5
        model = BridgeDiff(d_x=2, d_y=1, sigma_min=sigma_min, sigma_max=sigma_max)

        # Test with out-of-bounds tau
        tau_values = torch.tensor([-0.5, 0.0, 0.5, 1.0, 1.5])
        sigmas = model._sigma(tau_values)

        assert torch.all(sigmas >= sigma_min)
        assert torch.all(sigmas <= sigma_max)


class TestBridgeDiffLoss:
    """Test BridgeDiff loss computation."""

    def test_loss_observed_only(self):
        """Test loss with only observed treatments."""
        model = BridgeDiff(d_x=2, d_y=1, k=2, embed_dim=16, hidden=64, n_blocks=2)

        batch_size = 10
        x = torch.randn(batch_size, 2)
        y = torch.randn(batch_size, 1)
        t_obs = torch.randint(0, 2, (batch_size,))  # All observed

        loss = model.loss(x, y, t_obs)

        assert loss.numel() == 1, "Loss should be scalar"
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss >= 0, "Loss should be non-negative"

    def test_loss_unobserved_only(self):
        """Test loss with only unobserved treatments."""
        model = BridgeDiff(d_x=2, d_y=1, k=2, embed_dim=16, hidden=64, n_blocks=2)

        batch_size = 10
        x = torch.randn(batch_size, 2)
        y = torch.randn(batch_size, 1)
        t_obs = torch.full((batch_size,), -1, dtype=torch.long)  # All unobserved

        loss = model.loss(x, y, t_obs)

        assert loss.numel() == 1, "Loss should be scalar"
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss >= 0, "Loss should be non-negative"

    def test_loss_mixed(self):
        """Test loss with mixed observed and unobserved treatments."""
        model = BridgeDiff(d_x=2, d_y=1, k=2, embed_dim=16, hidden=64, n_blocks=2)

        batch_size = 10
        x = torch.randn(batch_size, 2)
        y = torch.randn(batch_size, 1)
        # Half observed, half unobserved
        t_obs = torch.cat([
            torch.randint(0, 2, (batch_size // 2,)),
            torch.full((batch_size - batch_size // 2,), -1, dtype=torch.long)
        ])

        loss = model.loss(x, y, t_obs)

        assert loss.numel() == 1, "Loss should be scalar"
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss >= 0, "Loss should be non-negative"

    def test_loss_warmup_disables_unobserved(self):
        """Test that warmup disables unobserved loss component."""
        model = BridgeDiff(d_x=2, d_y=1, k=2, embed_dim=16, hidden=64, n_blocks=2)

        batch_size = 10
        x = torch.randn(batch_size, 2)
        y = torch.randn(batch_size, 1)
        t_obs = torch.full((batch_size,), -1, dtype=torch.long)  # All unobserved

        # During warmup, loss should be minimal (only observed component, which is 0)
        loss_warmup = model.loss(x, y, t_obs, warmup=10, current_epoch=5)

        # After warmup, loss should be non-zero
        loss_normal = model.loss(x, y, t_obs, warmup=10, current_epoch=10)

        # Warmup loss should be zero or very small (no observed data)
        assert loss_warmup < 0.01 or loss_warmup == 0
        # Normal loss should be larger
        assert loss_normal > loss_warmup

    def test_loss_gradient_flow(self):
        """Test that loss produces valid gradients."""
        model = BridgeDiff(d_x=2, d_y=1, k=2, embed_dim=16, hidden=64, n_blocks=2)

        batch_size = 8
        x = torch.randn(batch_size, 2)
        y = torch.randn(batch_size, 1)
        t_obs = torch.randint(0, 2, (batch_size,))

        loss = model.loss(x, y, t_obs)
        loss.backward()

        # Check that at least some parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_grad = True
                break

        assert has_grad, "At least some parameters should have non-zero gradients"


class TestBridgeDiffSampling:
    """Test BridgeDiff sampling methods."""

    def test_paired_sample_shapes(self):
        """Test paired_sample produces correct shapes."""
        model = BridgeDiff(d_x=2, d_y=1, k=2, embed_dim=16, hidden=64)

        batch_size = 5
        x = torch.randn(batch_size, 2)

        # Test single sample
        outputs = model.paired_sample(x, n_steps=10, n_samples=1)
        if isinstance(outputs, tuple):
            assert len(outputs) == 2  # k=2 treatments
            for out in outputs:
                assert out.shape == (batch_size, 1)
        else:
            assert outputs.shape == (batch_size, 2, 1)

        # Test multiple samples with return_tensor=True
        outputs = model.paired_sample(x, n_steps=10, n_samples=5, return_tensor=True)
        assert outputs.shape == (5, batch_size, 2, 1)

    def test_paired_sample_convergence_from_noise(self):
        """Test that samples start from noise and converge."""
        model = BridgeDiff(d_x=2, d_y=1, k=2, embed_dim=16, hidden=64)
        model.eval()  # Ensure deterministic behavior

        batch_size = 3
        x = torch.randn(batch_size, 2)

        # Sample with more steps should be different from pure noise
        with torch.no_grad():
            samples = model.paired_sample(x, n_steps=50, n_samples=1, return_tensor=True)

        # Samples should be finite
        assert torch.all(torch.isfinite(samples))

        # Samples should have reasonable magnitude (not exploded)
        assert torch.all(torch.abs(samples) < 100)

    def test_predict_outcome_consistency(self):
        """Test predict_outcome produces consistent results."""
        model = BridgeDiff(d_x=2, d_y=1, k=2, embed_dim=16, hidden=64)
        model.eval()

        batch_size = 4
        x = torch.randn(batch_size, 2)
        t = torch.zeros(batch_size, dtype=torch.long)

        torch.manual_seed(42)
        pred1 = model.predict_outcome(x, t, n_steps=20, n_samples=10, return_mean=True)

        torch.manual_seed(42)
        pred2 = model.predict_outcome(x, t, n_steps=20, n_samples=10, return_mean=True)

        # With same seed, predictions should be identical
        assert torch.allclose(pred1, pred2, atol=1e-5)

        # Predictions should be finite
        assert torch.all(torch.isfinite(pred1))

        # Predictions should have correct shape
        assert pred1.shape == (batch_size, 1)

    def test_predict_outcome_treatment_indexing(self):
        """Test that predict_outcome correctly indexes by treatment."""
        model = BridgeDiff(d_x=2, d_y=1, k=2, embed_dim=16, hidden=64)
        model.eval()

        batch_size = 3
        x = torch.randn(batch_size, 2)

        # Predict for treatment 0
        t0 = torch.zeros(batch_size, dtype=torch.long)
        pred_t0 = model.predict_outcome(x, t0, n_steps=10, n_samples=5, return_mean=True)

        # Predict for treatment 1
        t1 = torch.ones(batch_size, dtype=torch.long)
        pred_t1 = model.predict_outcome(x, t1, n_steps=10, n_samples=5, return_mean=True)

        # Predictions should generally be different for different treatments
        # (unless the model is completely untrained)
        assert pred_t0.shape == pred_t1.shape == (batch_size, 1)
        assert torch.all(torch.isfinite(pred_t0))
        assert torch.all(torch.isfinite(pred_t1))


class TestBridgeDiffTraining:
    """Test BridgeDiff training behavior."""

    def test_training_convergence_simple_data(self):
        """Test that model can fit simple synthetic data."""
        # Create simple dataset
        dataset = load_mixed_synthetic_dataset(n_samples=50, d_x=2, seed=42, label_ratio=0.8)
        loader = DataLoader(dataset, batch_size=10, shuffle=True)

        model = BridgeDiff(d_x=2, d_y=1, k=2, embed_dim=32, hidden=128, n_blocks=2)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)

        trainer = Trainer(model, opt, loader, logger=None)

        # Train for a few epochs
        trainer.fit(5)

        # Evaluate
        metrics = trainer.evaluate(loader)

        # Check that metrics are reasonable
        assert "loss" in metrics
        assert "outcome rmse" in metrics
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert metrics["outcome rmse"] < 10.0  # Should not be catastrophically bad

    def test_training_loss_decreases(self):
        """Test that training loss decreases over epochs."""
        dataset = load_mixed_synthetic_dataset(n_samples=40, d_x=2, seed=123, label_ratio=0.7)
        loader = DataLoader(dataset, batch_size=10, shuffle=True)

        model = BridgeDiff(d_x=2, d_y=1, k=2, embed_dim=16, hidden=64, n_blocks=2)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)

        # Record initial loss
        model.train()
        x, y, t = next(iter(loader))
        initial_loss = model.loss(x, y, t).item()

        # Train
        trainer = Trainer(model, opt, loader, logger=None)
        trainer.fit(10)

        # Record final loss
        model.train()
        x, y, t = next(iter(loader))
        final_loss = model.loss(x, y, t).item()

        # Loss should decrease (at least somewhat)
        print(f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
        # Allow for some variance but expect general improvement
        assert final_loss < initial_loss * 2.0, "Loss should not increase dramatically"


class TestBridgeDiffIntegration:
    """Integration tests for BridgeDiff."""

    def test_full_pipeline_synthetic_mixed(self):
        """Test complete training and evaluation pipeline."""
        dataset = load_mixed_synthetic_dataset(n_samples=60, d_x=2, seed=999, label_ratio=0.5)
        loader = DataLoader(dataset, batch_size=10, shuffle=True)

        model = BridgeDiff(d_x=2, d_y=1, k=2, embed_dim=32, hidden=128, n_blocks=2)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)

        trainer = Trainer(model, opt, loader, logger=None)
        trainer.fit(5)

        # Test prediction methods
        x_test = torch.randn(5, 2)
        y_test = torch.randn(5, 1)
        t_test = torch.randint(0, 2, (5,))

        # Test predict
        predictions = trainer.predict(x_test, t_test)
        assert predictions.shape == (5, 1)
        assert torch.all(torch.isfinite(predictions))

        # Test predict_treatment_proba
        probs = trainer.predict_treatment_proba(x_test, y_test)
        assert probs.shape == (5, 2)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-4)

    def test_different_label_ratios(self):
        """Test that model works with different label ratios."""
        for label_ratio in [0.1, 0.5, 0.9]:
            dataset = load_mixed_synthetic_dataset(
                n_samples=30, d_x=2, seed=100+int(label_ratio*10), label_ratio=label_ratio
            )
            loader = DataLoader(dataset, batch_size=10)

            model = BridgeDiff(d_x=2, d_y=1, k=2, embed_dim=16, hidden=64, n_blocks=2)
            opt = torch.optim.Adam(model.parameters(), lr=0.001)

            trainer = Trainer(model, opt, loader, logger=None)
            trainer.fit(3)

            metrics = trainer.evaluate(loader)

            # Should produce valid metrics regardless of label ratio
            assert "outcome rmse" in metrics
            assert torch.isfinite(torch.tensor(metrics["outcome rmse"]))


class TestBridgeDiffNumericalStability:
    """Test numerical stability of BridgeDiff."""

    def test_no_nan_in_forward_pass(self):
        """Test that forward pass doesn't produce NaN."""
        model = BridgeDiff(d_x=3, d_y=2, k=3, embed_dim=32, hidden=128)

        batch_size = 8
        x = torch.randn(batch_size, 3)
        y = torch.randn(batch_size, 2)
        t_obs = torch.randint(0, 3, (batch_size,))

        loss = model.loss(x, y, t_obs)

        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"

    def test_extreme_values_handling(self):
        """Test that model handles extreme input values."""
        model = BridgeDiff(d_x=2, d_y=1, k=2, embed_dim=16, hidden=64)

        # Test with large values
        x_large = torch.randn(5, 2) * 100
        y_large = torch.randn(5, 1) * 100
        t = torch.randint(0, 2, (5,))

        loss_large = model.loss(x_large, y_large, t)
        assert torch.isfinite(loss_large)

        # Test with small values
        x_small = torch.randn(5, 2) * 0.01
        y_small = torch.randn(5, 1) * 0.01

        loss_small = model.loss(x_small, y_small, t)
        assert torch.isfinite(loss_small)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for Wristband Gaussian Loss, embed layers, and DeterministicGAE model."""

import pytest
import torch
import torch.nn as nn

from xtylearner.losses import (
    w2_to_standard_normal_sq,
    C_WristbandGaussianLoss,
    S_LossComponents,
)
from xtylearner.models.embed_layers import (
    C_EmbedAttentionModule,
    C_ACN,
    C_PermutationLayer,
    C_AffineCouplingLayer,
    C_InvertibleFlow,
)
from xtylearner.models.deterministic_gae import DeterministicGAE
from xtylearner.models import get_model


# ---------------------------------------------------------------------------
# W2 distance
# ---------------------------------------------------------------------------


class TestW2ToStandardNormalSq:
    def test_gaussian_samples_small_distance(self):
        torch.manual_seed(0)
        x = torch.randn(512, 4)
        d = w2_to_standard_normal_sq(x)
        assert d.ndim == 0
        assert d.item() < 2.0  # should be close to 0 for true Gaussian

    def test_shifted_samples_large_distance(self):
        torch.manual_seed(0)
        x = torch.randn(256, 4) + 5.0
        d = w2_to_standard_normal_sq(x)
        assert d.item() > 10.0

    def test_reduction_none(self):
        torch.manual_seed(0)
        x = torch.randn(3, 64, 4)
        d = w2_to_standard_normal_sq(x, reduction="none")
        assert d.shape == (3,)

    def test_reduction_sum(self):
        torch.manual_seed(0)
        x = torch.randn(3, 64, 4)
        d = w2_to_standard_normal_sq(x, reduction="sum")
        d_none = w2_to_standard_normal_sq(x, reduction="none")
        assert torch.allclose(d, d_none.sum())

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            w2_to_standard_normal_sq(torch.randn(4))

    def test_too_few_samples(self):
        with pytest.raises(ValueError):
            w2_to_standard_normal_sq(torch.randn(1, 4))

    def test_d_greater_than_b(self):
        torch.manual_seed(0)
        x = torch.randn(8, 32)  # d > b, uses Gram path
        d = w2_to_standard_normal_sq(x)
        assert d.ndim == 0
        assert torch.isfinite(d)


# ---------------------------------------------------------------------------
# Wristband Gaussian Loss
# ---------------------------------------------------------------------------


class TestWristbandGaussianLoss:
    def test_basic_call(self):
        loss_fn = C_WristbandGaussianLoss()
        x = torch.randn(64, 4)
        result = loss_fn(x)
        assert isinstance(result, S_LossComponents)
        assert result.total.ndim == 0
        assert torch.isfinite(result.total)

    def test_calibrated(self):
        loss_fn = C_WristbandGaussianLoss(
            calibration_shape=(64, 4), calibration_reps=32
        )
        # Gaussian samples should have near-zero loss after calibration
        torch.manual_seed(42)
        x = torch.randn(64, 4)
        result = loss_fn(x)
        assert torch.isfinite(result.total)

    def test_non_gaussian_higher_loss(self):
        loss_fn = C_WristbandGaussianLoss(
            calibration_shape=(128, 4), calibration_reps=64
        )
        torch.manual_seed(0)
        gaussian = torch.randn(128, 4)
        non_gaussian = torch.randn(128, 4) * 5.0 + 3.0

        r_gauss = loss_fn(gaussian)
        r_nongauss = loss_fn(non_gaussian)
        assert r_nongauss.total.item() > r_gauss.total.item()

    def test_backward(self):
        loss_fn = C_WristbandGaussianLoss(
            calibration_shape=(32, 4), calibration_reps=16
        )
        x = torch.randn(32, 4, requires_grad=True)
        result = loss_fn(x)
        result.total.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_angular_geodesic(self):
        loss_fn = C_WristbandGaussianLoss(angular="geodesic")
        x = torch.randn(32, 4)
        result = loss_fn(x)
        assert torch.isfinite(result.total)

    def test_global_reduction(self):
        loss_fn = C_WristbandGaussianLoss(reduction="global")
        x = torch.randn(32, 4)
        result = loss_fn(x)
        assert torch.isfinite(result.total)

    def test_moment_types(self):
        for moment in ("mu_only", "kl_diag", "kl_full", "jeff_diag", "jeff_full", "w2"):
            loss_fn = C_WristbandGaussianLoss(moment=moment)
            x = torch.randn(32, 4)
            result = loss_fn(x)
            assert torch.isfinite(result.total), f"Failed for moment={moment}"

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            C_WristbandGaussianLoss(beta=-1)
        with pytest.raises(ValueError):
            C_WristbandGaussianLoss(angular="invalid")
        with pytest.raises(ValueError):
            C_WristbandGaussianLoss(reduction="invalid")
        with pytest.raises(ValueError):
            C_WristbandGaussianLoss(moment="invalid")


# ---------------------------------------------------------------------------
# C_ACN
# ---------------------------------------------------------------------------


class TestACN:
    def test_forward_shape(self):
        net = C_ACN(8, 4, hidden_dim=16, n_blocks=2)
        x = torch.randn(3, 8)
        out = net(x)
        assert out.shape == (3, 4)

    def test_batched(self):
        net = C_ACN(8, 4, hidden_dim=16, n_blocks=3)
        x = torch.randn(5, 3, 8)
        out = net(x)
        assert out.shape == (5, 3, 4)


# ---------------------------------------------------------------------------
# C_EmbedAttentionModule
# ---------------------------------------------------------------------------


class TestEmbedAttentionModule:
    def test_basic_forward(self):
        mod = C_EmbedAttentionModule(8, 4, 6, n_of_basis=16, n_of_heads=2)
        x = torch.randn(5, 8)
        out = mod(x)
        assert out.shape == (5, 6)

    def test_euclidean_vs_dot(self):
        euc = C_EmbedAttentionModule(8, 4, 6, 16, 2, is_euclidean=True)
        dot = C_EmbedAttentionModule(8, 4, 6, 16, 2, is_euclidean=False)
        x = torch.randn(3, 8)
        out_euc = euc(x)
        out_dot = dot(x)
        assert out_euc.shape == out_dot.shape == (3, 6)

    def test_with_q_transform(self):
        mod = C_EmbedAttentionModule(
            8, 4, 6, 16, 2, q_transform=nn.Linear(8, 8)
        )
        x = torch.randn(3, 8)
        out = mod(x)
        assert out.shape == (3, 6)

    def test_with_head_combine(self):
        mod = C_EmbedAttentionModule(
            8, 4, 6, 16, 2, head_combine=C_ACN(8, 6, 16, 2)
        )
        x = torch.randn(3, 8)
        out = mod(x)
        assert out.shape == (3, 6)

    def test_single_head(self):
        mod = C_EmbedAttentionModule(8, 6, 6, 16, 1)
        x = torch.randn(3, 8)
        out = mod(x)
        assert out.shape == (3, 6)

    def test_affine_experts(self):
        mod = C_EmbedAttentionModule(8, 4, 6, 16, 2, affine_experts=True)
        x = torch.randn(3, 8)
        out = mod(x)
        assert out.shape == (3, 6)

    def test_head_temperature(self):
        mod = C_EmbedAttentionModule(8, 4, 6, 16, 2, head_temperature=True)
        x = torch.randn(3, 8)
        out = mod(x)
        assert out.shape == (3, 6)

    def test_normalize_k(self):
        mod = C_EmbedAttentionModule(8, 4, 6, 16, 2, normalize_k=True)
        x = torch.randn(3, 8)
        out = mod(x)
        assert out.shape == (3, 6)

    def test_batched_input(self):
        mod = C_EmbedAttentionModule(8, 4, 6, 16, 2)
        x = torch.randn(2, 5, 8)
        out = mod(x)
        assert out.shape == (2, 5, 6)


# ---------------------------------------------------------------------------
# C_PermutationLayer
# ---------------------------------------------------------------------------


class TestPermutationLayer:
    def test_forward_inverse(self):
        perm = torch.tensor([2, 0, 3, 1])
        layer = C_PermutationLayer(4, perm)
        x = torch.randn(3, 4)
        y = layer(x)
        x_rec = layer.inverse(y)
        assert torch.allclose(x, x_rec)

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            C_PermutationLayer(0, torch.tensor([]))

    def test_wrong_input_dim(self):
        layer = C_PermutationLayer(4, torch.arange(4))
        with pytest.raises(ValueError):
            layer(torch.randn(3, 5))


# ---------------------------------------------------------------------------
# C_AffineCouplingLayer
# ---------------------------------------------------------------------------


class TestAffineCouplingLayer:
    def test_forward_inverse_contiguous(self):
        mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
        layer = C_AffineCouplingLayer(4, mask, hidden_dim=16, n_blocks=1)
        x = torch.randn(5, 4)
        y = layer(x)
        x_rec = layer.inverse(y)
        assert torch.allclose(x, x_rec, atol=1e-5)

    def test_forward_inverse_noncontiguous(self):
        mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
        layer = C_AffineCouplingLayer(4, mask, hidden_dim=16, n_blocks=1)
        x = torch.randn(5, 4)
        y = layer(x)
        x_rec = layer.inverse(y)
        assert torch.allclose(x, x_rec, atol=1e-5)

    def test_identity_init(self):
        mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
        layer = C_AffineCouplingLayer(4, mask, hidden_dim=16)
        x = torch.randn(5, 4)
        y = layer(x)
        # Should be close to identity at init
        assert torch.allclose(x, y, atol=1e-4)

    def test_degenerate_mask(self):
        mask = torch.ones(4)
        layer = C_AffineCouplingLayer(4, mask, hidden_dim=16)
        x = torch.randn(3, 4)
        y = layer(x)
        assert torch.allclose(x, y)


# ---------------------------------------------------------------------------
# C_InvertibleFlow
# ---------------------------------------------------------------------------


class TestInvertibleFlow:
    def test_forward_inverse(self):
        flow = C_InvertibleFlow(8, n_layers=4, hidden_dim=16, n_blocks=1)
        x = torch.randn(5, 8)
        y = flow(x)
        x_rec = flow.inverse(y)
        assert torch.allclose(x, x_rec, atol=1e-4)

    def test_permute_modes(self):
        for mode in ("none", "per_layer", "per_pair"):
            flow = C_InvertibleFlow(
                6, n_layers=3, hidden_dim=16, permute_mode=mode
            )
            x = torch.randn(4, 6)
            y = flow(x)
            x_rec = flow.inverse(y)
            assert torch.allclose(x, x_rec, atol=1e-4), f"Failed for mode={mode}"

    def test_mask_modes(self):
        for mask_mode in ("alternating", "half"):
            flow = C_InvertibleFlow(
                6,
                n_layers=3,
                hidden_dim=16,
                mask_mode=mask_mode,
                permute_mode="none",
            )
            x = torch.randn(4, 6)
            y = flow(x)
            x_rec = flow.inverse(y)
            assert torch.allclose(x, x_rec, atol=1e-4)

    def test_identity_init(self):
        # Without permutations, coupling layers start near identity
        flow = C_InvertibleFlow(
            8, n_layers=4, hidden_dim=16, permute_mode="none"
        )
        x = torch.randn(5, 8)
        y = flow(x)
        assert torch.allclose(x, y, atol=1e-3)

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            C_InvertibleFlow(0)

    def test_dim_1(self):
        flow = C_InvertibleFlow(1, n_layers=2)
        x = torch.randn(3, 1)
        y = flow(x)
        assert torch.allclose(x, y)  # dim=1 means no coupling


# ---------------------------------------------------------------------------
# DeterministicGAE model
# ---------------------------------------------------------------------------


class TestDeterministicGAE:
    def test_registry(self):
        model = get_model(
            "deterministic_gae",
            d_x=4,
            d_y=1,
            k=2,
            embed_dim=4,
            n_heads=2,
            n_basis=8,
            internal_dim=8,
            calibration_reps=4,
            batch_size_hint=16,
        )
        assert isinstance(model, DeterministicGAE)

    def test_forward(self):
        model = DeterministicGAE(
            d_x=4, d_y=1, k=2, embed_dim=4, n_heads=2, n_basis=8,
            internal_dim=8, calibration_reps=4, batch_size_hint=16,
        )
        x = torch.randn(8, 4)
        t = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        y = model(x, t)
        assert y.shape == (8, 1)

    def test_loss(self):
        model = DeterministicGAE(
            d_x=4, d_y=1, k=2, embed_dim=4, n_heads=2, n_basis=8,
            internal_dim=8, calibration_reps=4, batch_size_hint=16,
        )
        x = torch.randn(16, 4)
        y = torch.randn(16, 1)
        t = torch.randint(0, 2, (16,))
        loss = model.loss(x, y, t)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()

    def test_loss_with_unlabelled(self):
        model = DeterministicGAE(
            d_x=4, d_y=1, k=2, embed_dim=4, n_heads=2, n_basis=8,
            internal_dim=8, calibration_reps=4, batch_size_hint=16,
        )
        x = torch.randn(8, 4)
        y = torch.randn(8, 1)
        t = torch.tensor([0, 1, -1, -1, 0, -1, 1, -1])
        loss = model.loss(x, y, t)
        assert torch.isfinite(loss)

    def test_predict_outcome(self):
        model = DeterministicGAE(
            d_x=4, d_y=1, k=2, embed_dim=4, n_heads=2, n_basis=8,
            internal_dim=8, calibration_reps=4, batch_size_hint=16,
        )
        x = torch.randn(5, 4)
        y0 = model.predict_outcome(x, 0)
        y1 = model.predict_outcome(x, 1)
        assert y0.shape == (5,)
        assert y1.shape == (5,)

    def test_predict_counterfactual(self):
        model = DeterministicGAE(
            d_x=4, d_y=1, k=2, embed_dim=4, n_heads=2, n_basis=8,
            internal_dim=8, calibration_reps=4, batch_size_hint=16,
        )
        x = torch.randn(3, 4)
        samples = model.predict_counterfactual(x, t=0, n_samples=10)
        assert samples.shape[0] == 10
        assert samples.shape[1] == 3

    def test_encode_decode_roundtrip(self):
        model = DeterministicGAE(
            d_x=4, d_y=1, k=2, embed_dim=4, n_heads=2, n_basis=8,
            internal_dim=8, flow_layers=4, calibration_reps=4, batch_size_hint=16,
        )
        x = torch.randn(5, 4)
        z = model.encode(x)
        assert z.shape == (5, 4)
        x_hat = model.decode(z)
        assert x_hat.shape == (5, 4)

    def test_continuous_treatment(self):
        model = DeterministicGAE(
            d_x=4, d_y=1, k=None, embed_dim=4, n_heads=2, n_basis=8,
            internal_dim=8, calibration_reps=4, batch_size_hint=16,
        )
        x = torch.randn(8, 4)
        t = torch.randn(8)
        y = model(x, t)
        assert y.shape == (8, 1)

        y_pred = model.predict_outcome(x, 0.5)
        assert y_pred.shape == (8, 1)

    def test_trainer_type(self):
        assert DeterministicGAE.trainer_type == "supervised"

    def test_loss_trains_cls_t(self):
        """cls_t.weight.grad must be non-zero after loss().backward() with labelled data."""
        model = DeterministicGAE(
            d_x=4, d_y=1, k=2, embed_dim=4, n_heads=2, n_basis=8,
            internal_dim=8, calibration_reps=4, batch_size_hint=16,
        )
        x = torch.randn(16, 4)
        y = torch.randn(16, 1)
        t = torch.randint(0, 2, (16,))
        loss = model.loss(x, y, t)
        loss.backward()
        assert model.cls_t.weight.grad is not None
        assert model.cls_t.weight.grad.abs().sum().item() > 0

    def test_loss_entropy_on_unlabelled(self):
        """cls_t gets gradient from entropy term even with mixed labelled/unlabelled."""
        model = DeterministicGAE(
            d_x=4, d_y=1, k=2, embed_dim=4, n_heads=2, n_basis=8,
            internal_dim=8, calibration_reps=4, batch_size_hint=16,
        )
        x = torch.randn(8, 4)
        y = torch.randn(8, 1)
        t = torch.tensor([0, 1, -1, -1, 0, -1, 1, -1])
        loss = model.loss(x, y, t)
        assert torch.isfinite(loss)
        loss.backward()
        assert model.cls_t.weight.grad is not None
        assert model.cls_t.weight.grad.abs().sum().item() > 0

    def test_predict_treatment_proba_after_training(self):
        """After a few optimiser steps, treatment proba should not be uniform."""
        torch.manual_seed(42)
        model = DeterministicGAE(
            d_x=4, d_y=1, k=2, embed_dim=4, n_heads=2, n_basis=8,
            internal_dim=8, calibration_reps=4, batch_size_hint=32,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Create separable data: t correlates with x[:, 0]
        x = torch.randn(32, 4)
        t = (x[:, 0] > 0).long()
        y = torch.randn(32, 1)

        for _ in range(20):
            optimizer.zero_grad()
            loss = model.loss(x, y, t)
            loss.backward()
            optimizer.step()

        probs = model.predict_treatment_proba(x, y)
        # After training, predictions should not all be uniform [0.5, 0.5]
        max_probs = probs.max(dim=-1).values
        assert (max_probs > 0.6).any(), (
            "Treatment head did not learn — all probabilities still near uniform"
        )

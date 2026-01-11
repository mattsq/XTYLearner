"""Tests for ordinal classification components."""

import pytest
import torch

from xtylearner.losses import (
    coral_loss,
    coral_logits_to_class_probs,
    class_probs_to_cumulative,
    cumulative_link_loss,
    cumulative_to_class_probs,
    ordinal_focal_loss,
    ordinal_predict,
    ordinal_regression_loss,
)
from xtylearner.models.heads import (
    CORALHead,
    CumulativeLinkHead,
    OrdinalHead,
)
from xtylearner.training.metrics import (
    adjacent_accuracy,
    ordinal_accuracy,
    ordinal_mae,
    ordinal_metrics,
    ordinal_rmse,
    quadratic_weighted_kappa,
    spearman_correlation,
)


# ---------------------------------------------------------------------------
# Ordinal Loss Function Tests
# ---------------------------------------------------------------------------


class TestCumulativeLinkLoss:
    def test_basic(self):
        k = 4
        batch_size = 10
        # Cumulative probs should be monotonically increasing
        raw = torch.randn(batch_size, k - 1)
        cumprobs = torch.sigmoid(raw.sort(dim=-1).values)
        target = torch.randint(0, k, (batch_size,))

        loss = cumulative_link_loss(cumprobs, target, k)
        assert loss.shape == ()
        assert loss >= 0

    def test_perfect_prediction(self):
        k = 3
        # Target is class 1, cumprobs = [0.1, 0.9] -> P(T=1) = 0.8
        cumprobs = torch.tensor([[0.1, 0.9]])
        target = torch.tensor([1])

        loss = cumulative_link_loss(cumprobs, target, k)
        expected = -torch.log(torch.tensor(0.8))
        assert torch.allclose(loss, expected, atol=1e-5)


class TestCoralLoss:
    def test_basic(self):
        k = 5
        batch_size = 16
        logits = torch.randn(batch_size, k - 1)
        target = torch.randint(0, k, (batch_size,))

        loss = coral_loss(logits, target, k)
        assert loss.shape == ()
        assert loss >= 0

    def test_binary_targets(self):
        k = 4
        # For target=2, binary targets should be [1, 1, 0] (T>0, T>1, but not T>2)
        logits = torch.zeros(1, k - 1)
        target = torch.tensor([2])

        loss = coral_loss(logits, target, k)
        # With logits=0, sigmoid=0.5, BCE for [1,1,0] targets
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.tensor([[1.0, 1.0, 0.0]])
        )
        assert torch.allclose(loss, expected)


class TestOrdinalRegressionLoss:
    def test_basic(self):
        logits = torch.randn(10, 5)
        target = torch.randint(0, 5, (10,))

        loss = ordinal_regression_loss(logits, target, alpha=1.0)
        assert loss.shape == ()
        assert loss >= 0

    def test_soft_labels(self):
        k = 5
        logits = torch.zeros(1, k)
        target = torch.tensor([2])

        # With uniform logits, loss depends on soft labels
        loss = ordinal_regression_loss(logits, target, alpha=1.0)
        assert loss > 0


class TestOrdinalFocalLoss:
    def test_basic(self):
        logits = torch.randn(10, 5)
        target = torch.randint(0, 5, (10,))

        loss = ordinal_focal_loss(logits, target, gamma=2.0, alpha=1.0)
        assert loss.shape == ()
        assert loss >= 0

    def test_gamma_effect(self):
        # Higher gamma should increase the focal effect
        logits = torch.tensor([[1.0, 0.0, 0.0]])
        target = torch.tensor([0])

        loss_gamma0 = ordinal_focal_loss(logits, target, gamma=0.0, alpha=1.0)
        loss_gamma2 = ordinal_focal_loss(logits, target, gamma=2.0, alpha=1.0)

        # With gamma > 0, focal weighting reduces loss for well-classified examples
        # The relationship depends on the confidence level
        assert loss_gamma0 != loss_gamma2


# ---------------------------------------------------------------------------
# Probability Conversion Tests
# ---------------------------------------------------------------------------


class TestCumulativeToClassProbs:
    def test_basic_conversion(self):
        # P(T<=0)=0.2, P(T<=1)=0.7 -> P(T=0)=0.2, P(T=1)=0.5, P(T=2)=0.3
        cumprobs = torch.tensor([[0.2, 0.7]])
        class_probs = cumulative_to_class_probs(cumprobs)

        expected = torch.tensor([[0.2, 0.5, 0.3]])
        assert torch.allclose(class_probs, expected, atol=1e-5)

    def test_sums_to_one(self):
        cumprobs = torch.rand(5, 4).sort(dim=-1).values
        class_probs = cumulative_to_class_probs(cumprobs)

        assert torch.allclose(class_probs.sum(dim=-1), torch.ones(5), atol=1e-5)


class TestClassProbsToCumulative:
    def test_roundtrip(self):
        k = 5
        probs = torch.softmax(torch.randn(3, k), dim=-1)
        cumprobs = class_probs_to_cumulative(probs)
        recovered = cumulative_to_class_probs(cumprobs)

        assert torch.allclose(probs, recovered, atol=1e-5)


class TestCoralLogitsToClassProbs:
    def test_basic(self):
        # logits for P(T>j), all zeros -> sigmoid=0.5
        logits = torch.zeros(1, 3)  # k=4
        probs = coral_logits_to_class_probs(logits)

        # P(T>0)=0.5, P(T>1)=0.5, P(T>2)=0.5
        # P(T=0)=1-0.5=0.5, P(T=1)=0.5-0.5=0, P(T=2)=0.5-0.5=0, P(T=3)=0.5
        # But with clamp(min=0), we get approximately [0.5, 0, 0, 0.5]
        assert probs.shape == (1, 4)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)

    def test_ordered_logits(self):
        # High positive logits mean high P(T>j), concentrating mass in middle-high
        # Low negative logits mean low P(T>j), concentrating mass in lower classes
        logits_high = torch.tensor([[5.0, 5.0, 5.0]])  # k=4, all P(T>j) near 1
        logits_low = torch.tensor([[-5.0, -5.0, -5.0]])  # k=4, all P(T>j) near 0

        probs_high = coral_logits_to_class_probs(logits_high)
        probs_low = coral_logits_to_class_probs(logits_low)

        assert probs_high.shape == (1, 4)
        assert probs_low.shape == (1, 4)
        # With all P(T>j) near 1, mass concentrates on highest class
        assert probs_high[0, 3] > probs_high[0, 0]
        # With all P(T>j) near 0, mass concentrates on lowest class
        assert probs_low[0, 0] > probs_low[0, 3]


class TestOrdinalPredict:
    def test_mode_logits(self):
        logits = torch.tensor([[1.0, 2.0, 0.5]])  # Class 1 is most likely
        pred = ordinal_predict(logits, method="mode", output_type="logits")
        assert pred.item() == 1

    def test_mean_logits(self):
        # Uniform-ish distribution
        logits = torch.zeros(1, 5)
        pred = ordinal_predict(logits, method="mean", output_type="logits")
        assert pred.item() == 2  # E[T] = 2 for uniform over {0,1,2,3,4}

    def test_median_logits(self):
        logits = torch.zeros(1, 5)
        pred = ordinal_predict(logits, method="median", output_type="logits")
        assert pred.item() == 2  # Median of {0,1,2,3,4}


# ---------------------------------------------------------------------------
# Ordinal Head Tests
# ---------------------------------------------------------------------------


class TestCumulativeLinkHead:
    def test_forward_shape(self):
        head = CumulativeLinkHead(in_features=10, k=5)
        x = torch.randn(8, 10)
        cumprobs = head(x)

        assert cumprobs.shape == (8, 4)  # k-1 cumulative probs

    def test_cumprobs_monotonic(self):
        head = CumulativeLinkHead(in_features=10, k=5)
        x = torch.randn(8, 10)
        cumprobs = head(x)

        # Cumulative probs should be monotonically increasing
        for i in range(cumprobs.size(1) - 1):
            assert (cumprobs[:, i] <= cumprobs[:, i + 1] + 1e-5).all()

    def test_predict_proba_shape(self):
        head = CumulativeLinkHead(in_features=10, k=5)
        x = torch.randn(8, 10)
        probs = head.predict_proba(x)

        assert probs.shape == (8, 5)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)

    def test_k_equals_2(self):
        head = CumulativeLinkHead(in_features=5, k=2)
        x = torch.randn(4, 5)
        cumprobs = head(x)
        probs = head.predict_proba(x)

        assert cumprobs.shape == (4, 1)
        assert probs.shape == (4, 2)


class TestCORALHead:
    def test_forward_shape(self):
        head = CORALHead(in_features=10, k=5)
        x = torch.randn(8, 10)
        logits = head(x)

        assert logits.shape == (8, 4)  # k-1 binary logits

    def test_predict_proba_shape(self):
        head = CORALHead(in_features=10, k=5)
        x = torch.randn(8, 10)
        probs = head.predict_proba(x)

        assert probs.shape == (8, 5)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)

    def test_weight_sharing(self):
        head = CORALHead(in_features=10, k=5)
        # The fc layer should have no bias (weight sharing via separate biases)
        assert head.fc.bias is None
        assert head.biases.shape == (4,)


class TestOrdinalHead:
    def test_cumulative_method(self):
        head = OrdinalHead(in_features=10, k=5, method="cumulative")
        x = torch.randn(8, 10)

        output = head(x)
        probs = head.predict_proba(x)
        preds = head.predict(x)

        assert output.shape == (8, 4)
        assert probs.shape == (8, 5)
        assert preds.shape == (8,)

    def test_coral_method(self):
        head = OrdinalHead(in_features=10, k=5, method="coral")
        x = torch.randn(8, 10)

        output = head(x)
        probs = head.predict_proba(x)

        assert output.shape == (8, 4)
        assert probs.shape == (8, 5)

    def test_standard_method(self):
        head = OrdinalHead(in_features=10, k=5, method="standard")
        x = torch.randn(8, 10)

        output = head(x)
        probs = head.predict_proba(x)

        assert output.shape == (8, 5)  # Standard softmax
        assert probs.shape == (8, 5)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            OrdinalHead(in_features=10, k=5, method="invalid")


# ---------------------------------------------------------------------------
# Ordinal Metrics Tests
# ---------------------------------------------------------------------------


class TestOrdinalMAE:
    def test_perfect_prediction(self):
        preds = torch.tensor([0, 1, 2, 3])
        targets = torch.tensor([0, 1, 2, 3])
        mae = ordinal_mae(preds, targets)
        assert mae == 0.0

    def test_basic(self):
        preds = torch.tensor([0, 1, 3])
        targets = torch.tensor([1, 1, 1])
        mae = ordinal_mae(preds, targets)
        assert mae == 1.0  # |0-1| + |1-1| + |3-1| = 1 + 0 + 2 = 3, /3 = 1


class TestOrdinalRMSE:
    def test_perfect_prediction(self):
        preds = torch.tensor([0, 1, 2, 3])
        targets = torch.tensor([0, 1, 2, 3])
        rmse = ordinal_rmse(preds, targets)
        assert rmse == 0.0

    def test_basic(self):
        preds = torch.tensor([0, 2])
        targets = torch.tensor([1, 1])
        rmse = ordinal_rmse(preds, targets)
        # (1^2 + 1^2) / 2 = 1, sqrt(1) = 1
        assert torch.isclose(rmse, torch.tensor(1.0))


class TestOrdinalAccuracy:
    def test_exact_match(self):
        preds = torch.tensor([0, 1, 2, 3])
        targets = torch.tensor([0, 1, 2, 3])
        acc = ordinal_accuracy(preds, targets, tolerance=0)
        assert acc == 1.0

    def test_with_tolerance(self):
        preds = torch.tensor([0, 1, 2, 3])
        targets = torch.tensor([1, 1, 1, 1])
        # Off by: 1, 0, 1, 2
        acc_0 = ordinal_accuracy(preds, targets, tolerance=0)
        acc_1 = ordinal_accuracy(preds, targets, tolerance=1)
        acc_2 = ordinal_accuracy(preds, targets, tolerance=2)

        assert acc_0 == 0.25  # Only class 1 matches
        assert acc_1 == 0.75  # Classes 0, 1, 2 within tolerance
        assert acc_2 == 1.0  # All within tolerance


class TestAdjacentAccuracy:
    def test_basic(self):
        preds = torch.tensor([0, 1, 2, 3])
        targets = torch.tensor([1, 1, 1, 1])
        acc = adjacent_accuracy(preds, targets)
        assert acc == 0.75  # 0, 1, 2 are within 1 of target 1


class TestQuadraticWeightedKappa:
    def test_perfect_agreement(self):
        preds = torch.tensor([0, 1, 2, 3, 4])
        targets = torch.tensor([0, 1, 2, 3, 4])
        kappa = quadratic_weighted_kappa(preds, targets, k=5)
        assert torch.isclose(kappa, torch.tensor(1.0))

    def test_random_agreement(self):
        # Near-zero kappa for random predictions
        torch.manual_seed(42)
        preds = torch.randint(0, 5, (100,))
        targets = torch.randint(0, 5, (100,))
        kappa = quadratic_weighted_kappa(preds, targets, k=5)
        assert abs(kappa) < 0.5  # Should be close to 0

    def test_worst_case(self):
        # Systematically wrong: predict 0 when 4, 4 when 0
        preds = torch.tensor([4, 4, 0, 0])
        targets = torch.tensor([0, 0, 4, 4])
        kappa = quadratic_weighted_kappa(preds, targets, k=5)
        assert kappa < 0  # Negative kappa for worse-than-random


class TestSpearmanCorrelation:
    def test_perfect_positive(self):
        preds = torch.tensor([0, 1, 2, 3, 4])
        targets = torch.tensor([0, 1, 2, 3, 4])
        corr = spearman_correlation(preds, targets)
        assert torch.isclose(corr, torch.tensor(1.0))

    def test_perfect_negative(self):
        preds = torch.tensor([4, 3, 2, 1, 0])
        targets = torch.tensor([0, 1, 2, 3, 4])
        corr = spearman_correlation(preds, targets)
        assert torch.isclose(corr, torch.tensor(-1.0))

    def test_no_correlation(self):
        # Shuffled data should have low correlation
        torch.manual_seed(42)
        preds = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        targets = torch.tensor([5, 2, 9, 0, 7, 1, 8, 3, 6, 4])
        corr = spearman_correlation(preds, targets)
        assert abs(corr) < 0.5


class TestOrdinalMetrics:
    def test_all_metrics_returned(self):
        preds = torch.tensor([0, 1, 2, 3])
        targets = torch.tensor([0, 1, 2, 3])
        metrics = ordinal_metrics(preds, targets, k=4)

        assert "accuracy" in metrics
        assert "adjacent_accuracy" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "qwk" in metrics
        assert "spearman" in metrics

    def test_perfect_predictions(self):
        preds = torch.tensor([0, 1, 2, 3])
        targets = torch.tensor([0, 1, 2, 3])
        metrics = ordinal_metrics(preds, targets, k=4)

        assert metrics["accuracy"] == 1.0
        assert metrics["mae"] == 0.0
        assert torch.isclose(metrics["qwk"], torch.tensor(1.0))
        assert torch.isclose(metrics["spearman"], torch.tensor(1.0))


# ---------------------------------------------------------------------------
# Trainer Integration Tests
# ---------------------------------------------------------------------------


class TestTrainerIntegration:
    """Tests for ordinal classification integration with trainers."""

    def test_base_trainer_ordinal_metrics(self):
        """Test that BaseTrainer computes ordinal metrics for ordinal models."""
        from xtylearner.models import create_model
        from xtylearner.training import SupervisedTrainer

        # Create a simple ordinal model
        model = create_model(
            "dragon_net", d_x=5, d_y=1, k=4, ordinal=True, ordinal_method="coral"
        )

        # Create dummy data
        batch_size = 16
        x = torch.randn(batch_size, 5)
        y = torch.randn(batch_size, 1)
        t = torch.randint(0, 4, (batch_size,))

        # Create trainer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loader = [(x, y, t)]
        trainer = SupervisedTrainer(model, optimizer, train_loader)

        # Compute treatment metrics
        metrics = trainer._treatment_metrics(x, y, t)

        # Verify ordinal metrics are computed
        assert "treatment_mae" in metrics
        assert "treatment_qwk" in metrics
        assert "treatment_adjacent_acc" in metrics
        assert "treatment_accuracy" in metrics
        assert "nll" in metrics

    def test_base_trainer_non_ordinal_metrics(self):
        """Test that BaseTrainer uses standard metrics for non-ordinal models."""
        from xtylearner.models import create_model
        from xtylearner.training import SupervisedTrainer

        # Create a standard (non-ordinal) model
        model = create_model("dragon_net", d_x=5, d_y=1, k=4, ordinal=False)

        # Create dummy data
        batch_size = 16
        x = torch.randn(batch_size, 5)
        y = torch.randn(batch_size, 1)
        t = torch.randint(0, 4, (batch_size,))

        # Create trainer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loader = [(x, y, t)]
        trainer = SupervisedTrainer(model, optimizer, train_loader)

        # Compute treatment metrics
        metrics = trainer._treatment_metrics(x, y, t)

        # Verify standard metrics are computed (not ordinal metrics)
        assert "accuracy" in metrics
        assert "nll" in metrics
        assert "treatment_mae" not in metrics
        assert "treatment_qwk" not in metrics
        assert "treatment_adjacent_acc" not in metrics

    def test_ordinal_trainer_evaluate(self):
        """Test that OrdinalTrainer.evaluate() returns ordinal metrics."""
        from xtylearner.models import create_model
        from xtylearner.training import OrdinalTrainer

        # Create ordinal model
        model = create_model(
            "dragon_net", d_x=5, d_y=1, k=4, ordinal=True, ordinal_method="coral"
        )

        # Create dummy data
        batch_size = 16
        x = torch.randn(batch_size, 5)
        y = torch.randn(batch_size, 1)
        t = torch.randint(0, 4, (batch_size,))

        # Create trainer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loader = [(x, y, t)]
        val_loader = [(x, y, t)]
        trainer = OrdinalTrainer(model, optimizer, train_loader, val_loader)

        # Evaluate
        metrics = trainer.evaluate(val_loader)

        # Verify ordinal metrics are in the output
        assert "treatment mae" in metrics
        assert "treatment qwk" in metrics
        assert "treatment adjacent acc" in metrics
        assert "treatment accuracy" in metrics
        assert "loss" in metrics

    def test_ordinal_trainer_warning_non_ordinal(self):
        """Test that OrdinalTrainer warns when used with non-ordinal model."""
        import warnings
        from xtylearner.models import create_model
        from xtylearner.training import OrdinalTrainer

        # Create non-ordinal model
        model = create_model("dragon_net", d_x=5, d_y=1, k=4, ordinal=False)

        # Create trainer - should warn
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loader = [(torch.randn(16, 5), torch.randn(16, 1), torch.randint(0, 4, (16,)))]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trainer = OrdinalTrainer(model, optimizer, train_loader)
            assert len(w) == 1
            assert "ordinal=True" in str(w[0].message)

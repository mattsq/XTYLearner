import torch
from torch import nn
from torch.utils.data import TensorDataset

from xtylearner.active.label_propensity import (
    LabelPropensityTrainingConfig,
    train_label_propensity_model,
)
from xtylearner.active.strategies import DebiasedCoverageAcquisition


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()


class _DummyTrainer:
    def __init__(self, tau_need: torch.Tensor, propensity: torch.Tensor):
        self._tau_need = tau_need
        self._propensity = propensity

    def score_cate_need(self, X):
        return self._tau_need

    def predict_label_propensity(self, X):
        return self._propensity

    def get_calibrator(self):
        return None


def test_debiased_coverage_prefers_moderate_propensity():
    X = torch.randn(3, 2)
    tau = torch.tensor([1.0, 2.0, 0.5])
    propensity = torch.tensor([0.1, 0.5, 0.9])

    trainer = _DummyTrainer(tau, propensity)
    strategy = DebiasedCoverageAcquisition()
    strategy.set_trainer_context(trainer)

    scores = strategy(_DummyModel(), X, None, batch_size=2)
    expected = tau * propensity * (1 - propensity)
    assert torch.allclose(scores, expected)


def test_debiased_coverage_handles_degenerate_propensity():
    X = torch.randn(4, 2)
    tau = torch.tensor([3.0, 2.0, 1.0, 0.5])
    propensity = torch.ones(4)

    trainer = _DummyTrainer(tau, propensity)
    strategy = DebiasedCoverageAcquisition()
    strategy.set_trainer_context(trainer)

    scores = strategy(_DummyModel(), X, None, batch_size=2)
    assert torch.allclose(scores, tau)


def test_debiased_coverage_nan_safety():
    X = torch.randn(2, 2)
    tau = torch.tensor([float("nan"), 1.0])
    propensity = torch.tensor([0.5, 0.5])

    trainer = _DummyTrainer(tau, propensity)
    strategy = DebiasedCoverageAcquisition()
    strategy.set_trainer_context(trainer)

    scores = strategy(_DummyModel(), X, None, batch_size=1)
    assert torch.all(torch.isfinite(scores))
    assert scores[0] == 0.0


def test_train_label_propensity_model_learns_signal():
    torch.manual_seed(0)
    x_pos = torch.randn(32, 3)
    x_neg = torch.randn(32, 3) + 2.0

    labeled = TensorDataset(x_pos, torch.zeros(32, 1), torch.zeros(32, dtype=torch.long))
    unlabeled = TensorDataset(x_neg, torch.zeros(32, 1), torch.full((32,), -1, dtype=torch.long))

    config = LabelPropensityTrainingConfig(max_epochs=30, hidden_dim=16, lr=0.05)
    model = train_label_propensity_model(labeled, unlabeled, device="cpu", config=config)
    assert model is not None

    with torch.no_grad():
        pos_scores = model(x_pos).view(-1)
        neg_scores = model(x_neg).view(-1)

    assert pos_scores.mean() > neg_scores.mean()

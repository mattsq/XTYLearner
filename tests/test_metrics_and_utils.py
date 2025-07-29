import math
import torch
import pytest
from typing import Mapping

from xtylearner.training.metrics import (
    mse_loss,
    mae_loss,
    rmse_loss,
    cross_entropy_loss,
    accuracy,
)
from xtylearner.models.utils import ramp_up_sigmoid
from xtylearner.training.base_trainer import BaseTrainer
from xtylearner.models import GNN_SCM


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        probs = torch.tensor([[0.7, 0.3]], device=x.device)
        return probs.repeat(x.size(0), 1)

    def predict_outcome(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return 2 * x[:, :1]


class DummyContinuousModel(DummyModel):
    def __init__(self):
        super().__init__()
        self.k = None

    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        params = torch.tensor([[0.0, 1.0]], device=x.device)
        return params.repeat(x.size(0), 1)


class DummyTrainer(BaseTrainer):
    def fit(self, num_epochs: int) -> None:
        pass

    def evaluate(self, data_loader) -> Mapping[str, float]:
        return {"loss": 0.0, "treatment accuracy": 0.0, "outcome rmse": 0.0}

    def predict(self, *args, **kwargs):
        pass


def make_trainer():
    model = DummyModel()
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
    return DummyTrainer(model, opt, train_loader=[], device="cpu")


def make_continuous_trainer():
    model = DummyContinuousModel()
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
    return DummyTrainer(model, opt, train_loader=[], device="cpu")


def test_metrics_functions():
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([2.0, 0.0])

    assert mse_loss(pred, target).item() == pytest.approx(2.5)
    assert mae_loss(pred, target).item() == pytest.approx(1.5)
    assert rmse_loss(pred, target).item() == pytest.approx(math.sqrt(2.5))

    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1])
    ce = cross_entropy_loss(logits, labels)
    acc = accuracy(logits, labels)
    assert ce.item() == pytest.approx(0.126928, rel=1e-5)
    assert acc.item() == pytest.approx(1.0)


def test_ramp_up_sigmoid():
    assert ramp_up_sigmoid(0, 5) < 1.0
    assert ramp_up_sigmoid(5, 5) == pytest.approx(1.0)
    assert ramp_up_sigmoid(10, 5) == pytest.approx(1.0)


def test_extract_batch_with_and_without_treatment():
    trainer = make_trainer()
    x = torch.randn(3, 2)
    y = torch.randn(3, 1)
    out = trainer._extract_batch((x, y))
    assert out[0].shape == x.shape
    assert out[1].shape == y.shape
    assert torch.all(out[2] == -1)

    t = torch.tensor([0, 1, 0])
    out = trainer._extract_batch((x, y, t))
    assert torch.equal(out[2], t)


def test_metrics_from_loss():
    trainer = make_trainer()
    m1 = trainer._metrics_from_loss(torch.tensor(1.0))
    assert m1 == {"loss": 1.0}

    m2 = trainer._metrics_from_loss({"a": torch.tensor(2.0), "b": 3.0})
    assert m2 == {"a": 2.0, "b": 3.0}

    m3 = trainer._metrics_from_loss(4.0)
    assert m3 == {"loss": 4.0}


def test_treatment_metrics():
    trainer = make_trainer()
    x = torch.zeros(2, 1)
    y = torch.zeros(2, 1)
    t_obs = torch.tensor([0, 1])
    metrics = trainer._treatment_metrics(x, y, t_obs)
    expected_nll = torch.nn.functional.nll_loss(
        torch.tensor([[0.7, 0.3], [0.7, 0.3]]).log(), t_obs
    ).item()
    assert metrics["nll"] == pytest.approx(expected_nll)
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_outcome_metrics():
    trainer = make_trainer()
    x = torch.tensor([[1.0], [2.0]])
    y = 2 * x
    t_obs = torch.tensor([0, 1])
    metrics = trainer._outcome_metrics(x, y, t_obs)
    assert metrics["rmse"] == pytest.approx(0.0)

    metrics = trainer._outcome_metrics(x, y, torch.tensor([-1, -1]))
    assert metrics["rmse_unlabelled"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)


def test_extract_batch_continuous_model():
    trainer = make_continuous_trainer()
    x = torch.randn(2, 1)
    y = torch.randn(2, 1)
    _, _, t_obs = trainer._extract_batch((x, y))
    assert t_obs.dtype == torch.float32


def test_extract_batch_gnn_scm():
    model = GNN_SCM(d_x=2, d_y=1, k=None)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = DummyTrainer(model, opt, train_loader=[], device="cpu")
    x = torch.randn(2, 2)
    y = torch.randn(2, 1)
    _, _, t_obs = trainer._extract_batch((x, y))
    assert t_obs.dtype == torch.float32


def test_treatment_metrics_continuous():
    trainer = make_continuous_trainer()
    x = torch.zeros(2, 1)
    y = torch.zeros(2, 1)
    t_obs = torch.tensor([0.1, 0.2])
    metrics = trainer._treatment_metrics(x, y, t_obs)
    assert metrics == {}


class DummyHeadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k = None
        self.head_Y = torch.nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            self.head_Y.weight.copy_(torch.tensor([[1.0, 2.0, 3.0]]))


def test_predict_outcome_with_float_treatment():
    model = DummyHeadModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = DummyTrainer(model, opt, train_loader=[], device="cpu")
    x = torch.tensor([[1.0, 2.0]])
    t = torch.tensor([0.5])
    y = torch.zeros(1, 1)
    out = trainer._predict_outcome(x, t, y)
    assert out.item() == pytest.approx(6.5)


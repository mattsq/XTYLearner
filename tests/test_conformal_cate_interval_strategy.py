import torch
from torch.utils.data import TensorDataset

from xtylearner.active import ConformalCATEIntervalStrategy, ConformalCalibrator
from xtylearner.active.calibration import build_conformal_calibrator
from xtylearner.active.strategies import QueryStrategy


class _DummyBinaryModel(torch.nn.Module):
    def __init__(self, y0: torch.Tensor, y1: torch.Tensor) -> None:
        super().__init__()
        self.k = 2
        self.register_buffer("_y0", y0)
        self.register_buffer("_y1", y1)

    def predict_outcome(self, X: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() > 1:
            t = t.view(-1)
        return torch.where(t == 1, self._y1, self._y0)


class _OnesFallback(QueryStrategy):
    def forward(self, model, X_unlab, rep_fn, batch_size):
        return torch.ones(len(X_unlab))


def test_conformal_interval_strategy_prefers_widest_intervals():
    X = torch.zeros(3, 2)
    y0 = torch.tensor([0.1, 0.2, -0.3])
    y1 = torch.tensor([1.0, 0.4, 0.0])
    model = _DummyBinaryModel(y0, y1)

    calibrator = ConformalCalibrator(
        q_lo_t0=torch.tensor(-0.2),
        q_hi_t0=torch.tensor(0.3),
        q_lo_t1=torch.tensor(-0.1),
        q_hi_t1=torch.tensor(0.4),
    )

    strategy = ConformalCATEIntervalStrategy()
    strategy.update_calibrator(calibrator)
    scores = strategy(model, X, None, batch_size=2)

    expected_lo1 = y1 + calibrator.q_lo_t1
    expected_hi1 = y1 + calibrator.q_hi_t1
    expected_lo0 = y0 + calibrator.q_lo_t0
    expected_hi0 = y0 + calibrator.q_hi_t0

    tau_lo = expected_lo1 - expected_hi0
    tau_hi = expected_hi1 - expected_lo0
    expected_width = torch.clamp(tau_hi - tau_lo, min=0.0)

    assert torch.allclose(scores, expected_width)
    assert scores.argmax().item() == expected_width.argmax().item()


def test_conformal_interval_strategy_falls_back_without_calibrator():
    X = torch.zeros(2, 1)
    y0 = torch.tensor([0.0, 0.0])
    y1 = torch.tensor([1.0, 1.0])
    model = _DummyBinaryModel(y0, y1)

    fallback = _OnesFallback()
    strategy = ConformalCATEIntervalStrategy(fallback=fallback)
    scores = strategy(model, X, None, batch_size=2)

    assert torch.allclose(scores, torch.ones_like(scores))


def test_conformal_interval_strategy_clamps_negative_width():
    X = torch.zeros(1, 2)
    model = _DummyBinaryModel(torch.tensor([0.0]), torch.tensor([0.0]))

    calibrator = ConformalCalibrator(
        q_lo_t0=torch.tensor(1.0),
        q_hi_t0=torch.tensor(-0.5),
        q_lo_t1=torch.tensor(0.8),
        q_hi_t1=torch.tensor(-0.3),
    )

    strategy = ConformalCATEIntervalStrategy()
    strategy.update_calibrator(calibrator)
    scores = strategy(model, X, None, batch_size=1)

    assert torch.all(scores >= 0)
    assert torch.allclose(scores, torch.zeros_like(scores))


def test_build_conformal_calibrator_estimates_quantiles():
    class ResidualModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.k = 2

        def predict_outcome(self, X: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            if not torch.is_tensor(t):
                t = torch.tensor([t], device=X.device)
            if t.dim() > 1:
                t = t.view(-1)
            if t.numel() == 1:
                t = t.expand(len(X))
            return torch.zeros(len(X), device=X.device)

    model = ResidualModel()
    X = torch.zeros(6, 1)
    T = torch.tensor([0, 0, 0, 1, 1, 1])
    Y = torch.tensor([0.0, 1.0, 2.0, -1.0, 0.0, 3.0])
    dataset = TensorDataset(X, Y, T)

    coverage = 0.8
    calibrator = build_conformal_calibrator(model, dataset, coverage=coverage, batch_size=2)

    assert calibrator is not None

    residuals_t0 = torch.tensor([0.0, 1.0, 2.0])
    residuals_t1 = torch.tensor([-1.0, 0.0, 3.0])
    lower_q = (1 - coverage) / 2
    upper_q = 1 - lower_q

    expected_lo_t0 = torch.quantile(residuals_t0, lower_q)
    expected_hi_t0 = torch.quantile(residuals_t0, upper_q)
    expected_lo_t1 = torch.quantile(residuals_t1, lower_q)
    expected_hi_t1 = torch.quantile(residuals_t1, upper_q)

    assert torch.isclose(calibrator.q_lo_t0, expected_lo_t0)
    assert torch.isclose(calibrator.q_hi_t0, expected_hi_t0)
    assert torch.isclose(calibrator.q_lo_t1, expected_lo_t1)
    assert torch.isclose(calibrator.q_hi_t1, expected_hi_t1)


def test_build_conformal_calibrator_handles_missing_arm():
    class ResidualModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.k = 2

        def predict_outcome(self, X: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            if not torch.is_tensor(t):
                t = torch.tensor([t], device=X.device)
            if t.dim() > 1:
                t = t.view(-1)
            if t.numel() == 1:
                t = t.expand(len(X))
            return torch.zeros(len(X), device=X.device)

    model = ResidualModel()
    X = torch.zeros(3, 1)
    T = torch.tensor([0, 0, 0])
    Y = torch.tensor([0.0, 1.0, -2.0])
    dataset = TensorDataset(X, Y, T)

    calibrator = build_conformal_calibrator(model, dataset, coverage=0.9, batch_size=2)
    assert calibrator is not None

    abs_max = torch.tensor([0.0, 1.0, -2.0]).abs().max()
    assert torch.isclose(calibrator.q_lo_t1, -abs_max)
    assert torch.isclose(calibrator.q_hi_t1, abs_max)

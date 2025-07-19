import torch
from xtylearner.models import MultiTask
from xtylearner.active import EntropyT, DeltaCATE, FCCMRadius


def test_entropy_strategy_runs():
    model = MultiTask(d_x=2, d_y=1, k=2)
    X = torch.randn(4, 2)
    strat = EntropyT()
    scores = strat(model, X, getattr(model, "h", None), 2)
    assert scores.shape == (4,)


def test_var_strategy_runs():
    model = MultiTask(d_x=2, d_y=1, k=2)
    X = torch.randn(3, 2)
    strat = DeltaCATE()
    scores = strat(model, X, getattr(model, "h", None), 2)
    assert scores.shape == (3,)


def test_fccm_strategy_combines():
    model = MultiTask(d_x=2, d_y=1, k=2)
    X_lab = torch.randn(2, 2)
    X_unlab = torch.randn(5, 2)
    strat = FCCMRadius()
    strat.update_labeled(X_lab)
    scores = strat(model, X_unlab, getattr(model, "h", None), 2)
    assert scores.shape == (5,)

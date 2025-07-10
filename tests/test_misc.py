import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from xtylearner.configs import load as load_config
from xtylearner.models.lt_flow_diff import _sigma
from xtylearner.training import ArrayTrainer


class DummyModel:
    def fit(self, *args, **kwargs):
        pass


def test_config_load(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("a: 1\nb:\n  c: 2\n")
    cfg = load_config(cfg_path)
    assert cfg == {"a": 1, "b": {"c": 2}}


def test_sigma_schedule_bounds():
    smin, smax = 0.1, 1.0
    assert _sigma(torch.tensor(0.0), smin, smax).item() == pytest.approx(smin)
    assert _sigma(torch.tensor(1.0), smin, smax).item() == pytest.approx(smax)


def test_collect_arrays_shapes():
    X = torch.randn(4, 2)
    Y = torch.randn(4, 1)
    T = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(X, Y, T), batch_size=2)
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
    trainer = ArrayTrainer(DummyModel(), opt, loader)
    X_arr, Y_arr, T_arr = trainer._collect_arrays(loader)
    assert X_arr.shape == (4, 2)
    assert Y_arr.shape == (4,)
    assert T_arr.shape == (4,)

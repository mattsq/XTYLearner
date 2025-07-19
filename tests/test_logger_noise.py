import torch
import pytest

from xtylearner.noise_schedules import add_noise
from xtylearner.training.logger import ConsoleLogger


def test_add_noise_deterministic():
    x = torch.zeros(2, 3)
    scale = torch.tensor(0.5)
    torch.manual_seed(0)
    expected = torch.randn_like(x) * scale + x
    torch.manual_seed(0)
    out = add_noise(x, scale)
    assert torch.allclose(out, expected)


def test_add_noise_vector_scale():
    x = torch.zeros(2, 2)
    scale = torch.tensor([0.0, 1.0])
    torch.manual_seed(1)
    noise = torch.randn_like(x)
    expected = x + noise * scale.unsqueeze(-1)
    torch.manual_seed(1)
    out = add_noise(x, scale)
    assert torch.allclose(out, expected)


def test_console_logger_outputs_and_averages(capsys):
    logger = ConsoleLogger(print_every=2)
    logger.start_epoch(1, 3)
    logger.log_step(1, 0, 3, {"loss": 1.0})
    logger.log_step(1, 1, 3, {"loss": 2.0})
    logger.log_step(1, 2, 3, {"loss": 3.0})
    logger.log_validation(1, {"val": 0.5})
    logger.end_epoch(1)
    out = capsys.readouterr().out
    assert "Epoch 1 [2/3]" in out
    assert "Epoch 1 [3/3]" in out
    assert "Epoch 1 validation: val=0.5000" in out
    assert "Epoch 1 finished:" in out
    assert logger.averages()["loss"] == pytest.approx(2.0)

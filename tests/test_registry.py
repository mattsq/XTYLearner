import pytest
import torch
from xtylearner.models import get_model, CycleDual


def test_get_model_valid():
    model = get_model("cycle_dual", d_x=2, d_y=1, k=2)
    assert isinstance(model, CycleDual)


def test_get_model_invalid():
    with pytest.raises(ValueError):
        get_model("unknown_model")

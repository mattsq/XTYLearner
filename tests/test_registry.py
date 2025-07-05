import pytest
import torch.nn as nn
from xtylearner.models import get_model, CycleDual


def test_get_model_valid():
    model = get_model("cycle_dual", d_x=2, d_y=1, k=2)
    assert isinstance(model, CycleDual)


def test_get_model_with_mlp_args():
    model = get_model(
        "cycle_dual",
        d_x=2,
        d_y=1,
        k=2,
        hidden_dims=[16],
        dropout=0.1,
    )
    layers = list(model.G_Y)
    assert isinstance(layers[1], nn.ReLU)
    assert any(isinstance(l, nn.Dropout) for l in layers)
    assert layers[0].out_features == 16


def test_get_model_invalid():
    with pytest.raises(ValueError):
        get_model("unknown_model")

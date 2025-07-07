import pytest
import torch.nn as nn
from xtylearner.models import (
    get_model,
    CycleDual,
    DiffusionCEVAE,
    EnergyDiffusionImputer,
    JointEBM,
    GFlowNetTreatment,
    EMModel,
)


def test_get_model_valid():
    model = get_model("cycle_dual", d_x=2, d_y=1, k=2)
    assert isinstance(model, CycleDual)

    model2 = get_model("diffusion_cevae", d_x=2, d_y=1, k=2)
    assert isinstance(model2, DiffusionCEVAE)

    model3 = get_model("eg_ddi", d_x=2, d_y=1)
    assert isinstance(model3, EnergyDiffusionImputer)

    model4 = get_model("joint_ebm", d_x=2, d_y=1)
    assert isinstance(model4, JointEBM)

    model5 = get_model("gflownet_treatment", d_x=2, d_y=1)
    assert isinstance(model5, GFlowNetTreatment)

    model6 = get_model("em", n_treatments=2)
    assert isinstance(model6, EMModel)


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
    assert any(isinstance(layer, nn.Dropout) for layer in layers)
    assert layers[0].out_features == 16


def test_get_model_invalid():
    with pytest.raises(ValueError):
        get_model("unknown_model")

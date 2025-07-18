import pytest
import torch.nn as nn
from xtylearner.models import (
    get_model,
    get_model_names,
    get_model_args,
    BridgeDiff,
    CycleDual,
    DiffusionCEVAE,
    EMModel,
    EnergyDiffusionImputer,
    GFlowNetTreatment,
    JSBF,
    JointEBM,
    GNN_EBM,
    DragonNet,
    LTFlowDiff,
    MaskedTabularTransformer,
    MixtureOfFlows,
    M2VAE,
    MultiTask,
    ProbCircuitModel,
    LP_KNN,
    MeanTeacher,
    VAT_Model,
    FixMatch,
    SSDMLModel,
    SemiITE,
    SS_CEVAE,
    CEVAE_M,
    GANITE,
    CTMT,
    SCGM,
)


@pytest.mark.parametrize(
    "name,cls,kwargs",
    [
        ("cycle_dual", CycleDual, {"d_x": 2, "d_y": 1, "k": 2}),
        ("diffusion_cevae", DiffusionCEVAE, {"d_x": 2, "d_y": 1, "k": 2}),
        ("eg_ddi", EnergyDiffusionImputer, {"d_x": 2, "d_y": 1}),
        ("joint_ebm", JointEBM, {"d_x": 2, "d_y": 1}),
        ("gnn_ebm", GNN_EBM, {"d_x": 2, "k_t": 1, "d_y": 1}),
        ("gflownet_treatment", GFlowNetTreatment, {"d_x": 2, "d_y": 1}),
        ("em", EMModel, {"k": 2}),
        ("flow_ssc", MixtureOfFlows, {"d_x": 2, "d_y": 1, "k": 2}),
        ("multitask", MultiTask, {"d_x": 2, "d_y": 1, "k": 2}),
        ("dragon_net", DragonNet, {"d_x": 2, "d_y": 1, "k": 2}),
        ("m2_vae", M2VAE, {"d_x": 2, "d_y": 1, "k": 2}),
        ("ss_cevae", SS_CEVAE, {"d_x": 2, "d_y": 1, "k": 2}),
        ("cevae_m", CEVAE_M, {"d_x": 2, "d_y": 1, "k": 2}),
        ("bridge_diff", BridgeDiff, {"d_x": 2, "d_y": 1, "embed_dim": 16}),
        ("lt_flow_diff", LTFlowDiff, {"d_x": 2, "d_y": 1}),
        ("jsbf", JSBF, {"d_x": 2, "d_y": 1}),
        ("masked_tabular_transformer", MaskedTabularTransformer, {"d_x": 2}),
        ("prob_circuit", ProbCircuitModel, {}),
        ("lp_knn", LP_KNN, {}),
        ("ganite", GANITE, {"d_x": 2, "d_y": 1}),
        (
            "mean_teacher",
            MeanTeacher,
            {"d_x": 3, "d_y": 1, "k": 2},
        ),
        ("vat", VAT_Model, {"d_x": 2, "d_y": 1, "k": 2}),
        ("fixmatch", FixMatch, {}),
        ("ss_dml", SSDMLModel, {}),
        ("semiite", SemiITE, {"d_x": 2, "d_y": 1}),
        ("ctm_t", CTMT, {"d_in": 4}),
        ("scgm", SCGM, {"d_x": 2, "d_y": 1, "k": 2}),
    ],
)
def test_get_model_valid(name, cls, kwargs):
    model = get_model(name, **kwargs)
    assert isinstance(model, cls)


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


def test_get_model_names():
    names = get_model_names()
    assert "cycle_dual" in names
    assert "em" in names


def test_get_model_args_valid():
    args = get_model_args("cycle_dual")
    assert "d_x" in args
    assert "hidden_dims" in args


def test_get_model_args_invalid():
    with pytest.raises(ValueError):
        get_model_args("unknown_model")

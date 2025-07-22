import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from xtylearner.data import load_synthetic_dataset
from xtylearner.models import CaCoRE
from xtylearner.training import Trainer


def test_cacore_shapes_and_trainer():
    ds = load_synthetic_dataset(n_samples=20, d_x=2, seed=0)
    X, Y, T = ds.tensors
    model = CaCoRE(d_x=2, d_y=1, k=2)
    loss = model.loss(X, Y, T)
    assert loss.dim() == 0
    rep = model(X[:5])
    assert rep.shape == (5, model.encoder[-1].out_features)
    out = model.predict_outcome(X, T)
    assert out.shape == (20, 1)

    loader = DataLoader(ds, batch_size=5)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)
    preds = trainer.predict(X[:5], T[:5])
    assert preds.shape == (5, 1)


def test_cacore_custom_mlp_args():
    model = CaCoRE(
        d_x=2,
        d_y=1,
        k=2,
        hidden_dims=[16],
        activation=nn.Tanh,
        dropout=0.1,
        norm_layer=nn.BatchNorm1d,
    )

    layers = list(model.encoder)
    assert isinstance(layers[1], nn.BatchNorm1d)
    assert isinstance(layers[2], nn.Tanh)
    assert any(isinstance(layer, nn.Dropout) for layer in layers)
    assert layers[0].out_features == 16

    out_layers = list(model.outcome_head)
    assert isinstance(out_layers[1], nn.BatchNorm1d)
    assert isinstance(out_layers[2], nn.Tanh)
    assert any(isinstance(layer, nn.Dropout) for layer in out_layers)

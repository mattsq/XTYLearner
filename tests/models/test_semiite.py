import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from xtylearner.data import load_synthetic_dataset
from xtylearner.models import SemiITE
from xtylearner.training import CoTrainTrainer
import torch.nn as nn


def test_semiite_shapes_and_trainer():
    ds = load_synthetic_dataset(n_samples=20, d_x=2, seed=0)
    X, Y, T = ds.tensors
    T_obs = T.clone()
    T_obs[::2] = -1

    model = SemiITE(d_x=2, d_y=1, k=2)
    rep = model.encode(X)
    mmd = model.compute_mmd(rep, T)
    assert mmd.dim() == 0
    out = model.predict_outcome(X, T)
    assert out.shape == (20, 1)

    loader = DataLoader(TensorDataset(X, Y, T_obs), batch_size=5)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = CoTrainTrainer(model, opt, loader)
    trainer.fit(1)
    metrics = trainer.evaluate(loader)
    assert set(metrics) >= {"loss", "treatment accuracy", "outcome rmse"}


def test_semiite_multiclass_trains():
    rng = np.random.default_rng(0)
    X = torch.from_numpy(rng.normal(size=(12, 2)).astype(np.float32))
    T_true = torch.from_numpy(rng.integers(0, 3, size=12).astype(np.int64))
    Y = torch.from_numpy((T_true.numpy() + rng.normal(size=12)).astype(np.float32)).unsqueeze(-1)
    T_obs = T_true.clone()
    T_obs[::3] = -1

    ds = TensorDataset(X, Y, T_obs)
    loader = DataLoader(ds, batch_size=4)
    model = SemiITE(d_x=2, d_y=1, k=3)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = CoTrainTrainer(model, opt, loader)
    trainer.fit(1)
    metrics = trainer.evaluate(loader)
    assert set(metrics) >= {"loss", "treatment accuracy", "outcome rmse"}


def test_semiite_custom_mlp_args():
    model = SemiITE(
        d_x=2,
        d_y=1,
        k=2,
        enc_hidden_dims=[16],
        outcome_hidden_dims=[8],
        activation=nn.Tanh,
        dropout=0.1,
        norm_layer=nn.BatchNorm1d,
    )
    layers = list(model.enc)
    assert isinstance(layers[1], nn.BatchNorm1d)
    assert isinstance(layers[2], nn.Tanh)
    assert any(isinstance(layer, nn.Dropout) for layer in layers)
    assert layers[0].out_features == 16

    out_layers = list(model.outcome[0])
    assert isinstance(out_layers[1], nn.BatchNorm1d)
    assert isinstance(out_layers[2], nn.Tanh)
    assert any(isinstance(layer, nn.Dropout) for layer in out_layers)
    assert out_layers[0].out_features == 8

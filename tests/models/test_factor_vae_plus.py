import torch
from torch.utils.data import TensorDataset, DataLoader

from xtylearner.data import load_synthetic_dataset
from xtylearner.models import FactorVAEPlus
from xtylearner.training import Trainer


def _get_dataset(n=20, d_x=2, seed=0):
    ds = load_synthetic_dataset(n_samples=n, d_x=d_x, seed=seed)
    X, Y, T = ds.tensors
    T = T.unsqueeze(1)
    return TensorDataset(X, Y, T)


def test_factor_vae_plus_shapes():
    ds = _get_dataset(n=10)
    X, Y, T = ds.tensors
    model = FactorVAEPlus(d_x=2, d_y=1, k=2)
    loss = model.elbo(X, Y, T)
    assert loss.dim() == 0
    out = model.predict_outcome(X, T)
    assert out.shape == (10, 1)
    probs = model.predict_treatment_proba(X, Y)
    assert probs[0].shape == (10, 2)


def test_factor_vae_plus_trainer():
    ds = _get_dataset(n=20, seed=1)
    loader = DataLoader(ds, batch_size=5)
    model = FactorVAEPlus(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)

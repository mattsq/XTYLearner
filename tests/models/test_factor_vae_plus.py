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


def test_factor_vae_plus_multiple_treatments():
    X = torch.randn(8, 2)
    Y = torch.randn(8, 1)
    T = torch.stack([
        torch.randint(0, 2, (8,), dtype=torch.long),
        torch.randint(0, 3, (8,), dtype=torch.long),
    ], dim=1)
    model = FactorVAEPlus(d_x=2, d_y=1, k=2, cat_sizes=[2, 3])
    loss = model.elbo(X, Y, T)
    assert loss.dim() == 0
    out = model.predict_outcome(X, T)
    assert out.shape == (8, 1)
    probs = model.predict_treatment_proba(X, Y)
    assert len(probs) == 2
    assert probs[0].shape == (8, 2)
    assert probs[1].shape == (8, 3)

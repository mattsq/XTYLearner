import torch
from torch.utils.data import DataLoader

from xtylearner.data import load_synthetic_dataset
from xtylearner.models import CEVAE_M
from xtylearner.training import Trainer


def test_shapes_and_loss_scalar():
    ds = load_synthetic_dataset(n_samples=10, d_x=2, seed=0)
    X, Y, T = ds.tensors
    model = CEVAE_M(d_x=2, d_y=1, k=2)
    loss = model.loss(X, Y, T)
    assert loss.dim() == 0
    probs = model.predict_treatment_proba(X, Y)
    assert probs.shape == (10, 2)
    out = model.predict_outcome(X, torch.zeros(10, dtype=torch.long))
    assert out.shape == (10, 1)


def test_backward_runs():
    ds = load_synthetic_dataset(n_samples=8, d_x=2, seed=1)
    X, Y, T = ds.tensors
    model = CEVAE_M(d_x=2, d_y=1, k=2)
    loss = model.loss(X, Y, T)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)


def test_trainer_integration():
    ds = load_synthetic_dataset(n_samples=20, d_x=2, seed=2)
    loader = DataLoader(ds, batch_size=5)
    model = CEVAE_M(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)


def test_pehe_sanity():
    ds = load_synthetic_dataset(n_samples=30, d_x=2, seed=3)
    X, Y, T = ds.tensors
    model = CEVAE_M(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loader = DataLoader(ds, batch_size=10)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)

    with torch.no_grad():
        y0 = model.predict_outcome(X, torch.zeros(len(X), dtype=torch.long))
        y1 = model.predict_outcome(X, torch.ones(len(X), dtype=torch.long))
    ite_pred = (y1 - y0).squeeze(1)
    true_ite = torch.full_like(ite_pred, 2.0)
    pehe = torch.sqrt(((ite_pred - true_ite) ** 2).mean())
    assert torch.isfinite(pehe)

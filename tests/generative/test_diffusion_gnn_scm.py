import torch
from torch.utils.data import DataLoader

from xtylearner.data import load_toy_dataset
from xtylearner.models import DiffusionGNN_SCM
from xtylearner.training import Trainer


def test_diffusion_gnn_scm_acyclic():
    ds = load_toy_dataset(n_samples=30, d_x=2, seed=1)
    loader = DataLoader(ds, batch_size=10)
    model = DiffusionGNN_SCM(d_x=2, k=2, d_y=1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    trainer.fit(0)
    A = torch.sigmoid(model.B.detach()) * model.mask
    acyc = float((torch.trace(torch.matrix_exp(A * A)) - A.size(0)))
    assert acyc < 0.01


def test_diffusion_gnn_scm_predict_and_proba():
    ds = load_toy_dataset(n_samples=20, d_x=2, seed=2)
    loader = DataLoader(ds, batch_size=10)
    model = DiffusionGNN_SCM(d_x=2, k=2, d_y=1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)

    X, Y, T = next(iter(loader))
    preds = trainer.predict(X, T)
    assert preds.shape == (10,)

    probs = trainer.predict_treatment_proba(X, Y)
    assert probs.shape == (10, 2)
    assert torch.allclose(probs.sum(-1), torch.ones(10), atol=1e-5)

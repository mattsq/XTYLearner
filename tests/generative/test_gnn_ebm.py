import torch
from torch.utils.data import DataLoader

from xtylearner.data import load_toy_dataset
from xtylearner.models import GNN_EBM
from xtylearner.training import Trainer


def test_gnn_ebm_backward():
    ds = load_toy_dataset(n_samples=20, d_x=2, seed=0)
    X, Y, T = ds.tensors
    model = GNN_EBM(d_x=2, k_t=1, d_y=1)
    loss = model.loss(X, Y, T)
    loss.backward()
    assert torch.isfinite(loss)


def test_gnn_ebm_trainer_runs():
    ds = load_toy_dataset(n_samples=20, d_x=2, seed=1)
    loader = DataLoader(ds, batch_size=5)
    model = GNN_EBM(d_x=2, k_t=1, d_y=1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)


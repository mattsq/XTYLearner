import torch
from torch.utils.data import DataLoader

from xtylearner.data import load_toy_dataset
from xtylearner.models import GNN_SCM
from xtylearner.training import Trainer


def test_gnn_scm_learns_acyclic():
    ds = load_toy_dataset(n_samples=50, d_x=2, seed=0)
    loader = DataLoader(ds, batch_size=10)
    model = GNN_SCM(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    A0 = torch.sigmoid(model.B.detach()) * model.mask
    init_acyc = float((torch.trace(torch.matrix_exp(A0 * A0)) - A0.size(0)))
    trainer.fit(4)
    A = torch.sigmoid(model.B.detach()) * model.mask
    final_acyc = float((torch.trace(torch.matrix_exp(A * A)) - A.size(0)))
    assert final_acyc <= init_acyc
    assert final_acyc < 1.0

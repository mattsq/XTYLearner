import torch
from torch.utils.data import TensorDataset, DataLoader

from xtylearner.data import load_mixed_synthetic_dataset
from xtylearner.models import SCGM
from xtylearner.training import Trainer


def _get_dataset(n=20, d_x=2, seed=0):
    ds = load_mixed_synthetic_dataset(n_samples=n, d_x=d_x, seed=seed, label_ratio=0.5)
    X, Y, T = ds.tensors
    Y = Y.clone()
    Y[::3] = float('nan')
    return TensorDataset(X, Y, T)


def test_scgm_loss_shapes():
    ds = _get_dataset(n=10)
    X, Y, T = ds.tensors
    model = SCGM(d_x=2, d_y=1, k=2)
    out = model.loss(X, Y, T)
    assert 'loss' in out and out['loss'].dim() == 0


def test_scgm_trainer_runs():
    ds = _get_dataset(n=20, seed=1)
    loader = DataLoader(ds, batch_size=5)
    model = SCGM(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)

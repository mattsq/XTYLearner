import torch
from torch.utils.data import DataLoader, TensorDataset

from xtylearner.data import load_synthetic_dataset
from xtylearner.models import SemiITE
from xtylearner.training import CoTrainTrainer


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

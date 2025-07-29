import torch
from xtylearner.data import load_mixed_synthetic_dataset
from xtylearner.models import CycleDual


def test_cycle_dual_continuous():
    ds = load_mixed_synthetic_dataset(
        n_samples=10, d_x=2, seed=0, label_ratio=0.5, continuous_treatment=True
    )
    X, Y, T_obs = ds.tensors
    model = CycleDual(d_x=2, d_y=1, k=None)
    loss = model.loss(X, Y, T_obs)
    assert loss.dim() == 0
    out = model.predict_outcome(X, torch.zeros(len(X)))
    assert out.shape == (len(X), 1)


import torch

from xtylearner.data import load_mixed_synthetic_dataset
from xtylearner.models import CycleVAT


def test_cycle_vat_runs_forward_and_loss():
    ds = load_mixed_synthetic_dataset(n_samples=10, d_x=2, seed=0, label_ratio=0.5)
    X, Y, T_obs = ds.tensors
    model = CycleVAT(d_x=2, d_y=1)

    loss = model.loss(X, Y, T_obs)
    assert loss.dim() == 0

    out = model.predict_outcome(X, torch.zeros(len(X), dtype=torch.long))
    assert out.shape == (len(X), 1)

    proba_x = model.predict_treatment_proba(X)
    assert proba_x.shape == (len(X), model.k)

    proba_xy = model.predict_treatment_proba(X, Y)
    assert proba_xy.shape == (len(X), model.k)

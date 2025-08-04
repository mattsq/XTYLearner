import torch
from xtylearner.data import load_mixed_synthetic_dataset
from xtylearner.models import CycleVAT


def test_cycle_vat_gradnorm_updates_weights():
    ds = load_mixed_synthetic_dataset(n_samples=10, d_x=2, seed=0, label_ratio=0.5)
    X, Y, T_obs = ds.tensors
    model = CycleVAT(d_x=2, d_y=1, gradnorm=True)
    w_before = model.loss_weights.detach().clone()
    loss = model.loss(X, Y, T_obs)
    assert loss.dim() == 0
    w_after = model.loss_weights.detach()
    assert not torch.allclose(w_before, w_after)
    out = model.predict_outcome(X, torch.zeros(len(X)))
    assert out.shape == (len(X), 1)

import torch
from xtylearner.models import LTFlowDiff


def test_flow_inverse_consistency():
    d_x, d_y, k = 3, 2, 2
    model = LTFlowDiff(d_x=d_x, d_y=d_y, k=k)
    x = torch.randn(16, d_x)
    y = torch.randn(16, d_y)
    z = torch.randn(16, model.d_z)
    t = torch.randint(0, k, (16,))
    u, ld = model.flow(y, x, z, t)
    y_rec, ld_inv = model.flow.inverse(u, x, z, t)
    assert torch.allclose(y, y_rec, atol=1e-4)
    assert ld.shape == ld_inv.shape


def test_score_output():
    model = LTFlowDiff(d_x=2, d_y=1)
    z = torch.randn(1, model.d_z)
    t = torch.zeros(1, dtype=torch.long)
    sig = model._sigma(torch.tensor([0.5]))
    eps = torch.randn_like(z)
    z_tau = z + sig * eps
    s = model.score(z_tau, t, tau=torch.tensor([[0.5]]))
    assert torch.isfinite(s).all()


import torch

from xtylearner.models import DeconfounderCFM
from xtylearner.models.deconfounder_model import _hsic


def generate_data(n=50, k_t=2, d_z=3):
    z = torch.randn(n, d_z)
    A = torch.randn(k_t, d_z)
    t = z @ A.t() + 0.1 * torch.randn(n, k_t)
    x = torch.zeros(n, 1)
    y = z.sum(1, keepdim=True) + t.sum(1, keepdim=True)
    return x, y, t


def test_shapes():
    x, y, t = generate_data()
    model = DeconfounderCFM(d_x=1, d_y=1, k_t=2, d_z=3, pretrain_epochs=1)
    loss = model.loss(x, y, t)
    assert loss.dim() == 0
    out = model.predict_outcome(x, t)
    assert out.shape == (x.size(0), 1)


def test_grad_flow():
    x, y, t = generate_data()
    model = DeconfounderCFM(d_x=1, d_y=1, k_t=2, d_z=3, pretrain_epochs=1)
    loss = model.loss(x, y, t)
    loss.backward()
    vae_grads = [p.grad.clone() for p in model.vae_t.parameters()]
    out_grads = [p.grad for p in model.out_net.parameters()]
    assert all(g is not None and g.abs().sum() > 0 for g in vae_grads)
    assert all(g is None for g in out_grads)
    for p in model.parameters():
        p.grad = None
    model.on_epoch_end()
    loss = model.loss(x, y, t)
    loss.backward()
    out_grads = [p.grad for p in model.out_net.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in out_grads)


def test_missing_t():
    x, y, t = generate_data()
    t[0, 0] = float("nan")
    model = DeconfounderCFM(d_x=1, d_y=1, k_t=2, d_z=3, pretrain_epochs=0)
    loss = model.loss(x, y, t)
    assert torch.isfinite(loss)


def test_ppc_hsic_small():
    x, y, t = generate_data(n=80)
    model = DeconfounderCFM(d_x=1, d_y=1, k_t=2, d_z=3, pretrain_epochs=5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(6):
        loss = model.loss(x, y, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z = model.vae_t.encode(t, sample=False)
        hsic = _hsic(t, z).item()
    assert hsic < 1.0

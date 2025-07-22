import torch
from xtylearner.models import CTMT


def test_ctmt_inferred_d_in_and_forward():
    model = CTMT(d_x=3, d_y=2, k=4)
    assert model.d_in == 3 + 2 + 1
    x = torch.randn(5, model.d_in)
    t = torch.zeros(5, 1)
    delta = torch.zeros(5, 1)
    out, logits = model(x, t, delta)
    assert out.shape == (5, model.d_in)
    assert logits.shape == (5, 4)

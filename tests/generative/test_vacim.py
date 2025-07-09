import torch
from xtylearner.models import get_model


def test_vacim_overfits_toy():
    x = torch.randn(256, 5)
    y = torch.randn(256, 1)
    t = torch.randint(0, 2, (256, 1)).float()
    t[::3] = float('nan')

    model = get_model('vacim', d_x=5, d_y=1, k=2)
    out = model.loss(x, y, t)
    out['loss'].backward()
    assert not torch.isnan(out['loss'])

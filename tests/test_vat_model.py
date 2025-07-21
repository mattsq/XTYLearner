import torch

from xtylearner.models import VAT_Model


def test_vat_basic_shapes():
    model = VAT_Model(d_x=3, d_y=1, k=2)
    x = torch.randn(5, 3)
    y = torch.randn(5, 1)
    t = torch.randint(0, 2, (5,))

    out = model(x, t)
    assert out.shape == (5, 1)

    loss = model.loss(x, y, t)
    assert isinstance(loss, torch.Tensor)

    probs = model.predict_treatment_proba(x, y)
    assert probs.shape == (5, 2)


def test_vat_predict_outcome():
    model = VAT_Model(d_x=3, d_y=1, k=2)
    x = torch.randn(4, 3)
    t = torch.randint(0, 2, (4,))

    out = model.predict_outcome(x, t)
    assert out.shape == (4, 1)

    out_scalar = model.predict_outcome(x, int(t[0].item()))
    assert out_scalar.shape == (4, 1)

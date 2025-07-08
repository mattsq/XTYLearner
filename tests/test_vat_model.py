import torch

from xtylearner.models import VAT_Model


def test_vat_supports_multitarget():
    X_lab = torch.randn(4, 3)
    y_lab = torch.randint(0, 2, (4, 2))
    X_unlab = torch.randn(6, 3)
    model = VAT_Model()
    model.fit(X_lab, y_lab, X_unlab, epochs=1, bs=2)
    out = model.predict_proba(torch.randn(3, 3))
    assert out.shape == (3, 2)

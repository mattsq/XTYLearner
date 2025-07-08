import torch

from xtylearner.models import VIME_Model


def test_vime_basic_training():
    X_lab = torch.randn(4, 3)
    y_lab = torch.randint(0, 2, (4, 2))
    X_unlab = torch.randn(6, 3)
    model = VIME_Model()
    model.fit(X_lab, y_lab, X_unlab, pre_epochs=1, pre_bs=2, sl_epochs=1, sl_bs=2)
    out = model.predict_proba(torch.randn(3, 3))
    assert out.shape == (3, 2)

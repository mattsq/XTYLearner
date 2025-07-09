import torch
from torch.utils.data import DataLoader, TensorDataset

from xtylearner.models import VIME
from xtylearner.training import Trainer


def test_vime_basic_training():
    X_lab = torch.randn(4, 3)
    y_lab = torch.randint(0, 2, (4,), dtype=torch.long)
    X_unlab = torch.randn(6, 3)

    model = VIME(d_x=3, d_y=2)

    X_train = torch.cat([X_lab, X_unlab])
    y_train = torch.cat([y_lab, torch.full((6,), -1)])
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=2, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)

    out = model.predict_proba(torch.randn(3, 3))
    assert out.shape == (3, 2)

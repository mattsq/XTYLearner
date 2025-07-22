import torch
from xtylearner.models import DragonNet


def test_predict_outcome_vector_t():
    model = DragonNet(d_x=3, d_y=2, k=3)
    x = torch.randn(4, 3)
    t = torch.tensor([0, 1, 2, 1])
    out = model.predict_outcome(x, t)
    assert out.shape == (4, 2)


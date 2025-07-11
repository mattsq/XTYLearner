import torch

from xtylearner.models import CNFlowModel


def test_shapes():
    d_x, d_y, k = 5, 2, 3
    x = torch.randn(17, d_x)
    y = torch.randn(17, d_y)
    t = torch.randint(0, k, (17,))
    mask = torch.randint(0, 2, (17,))
    t_obs = t.clone()
    t_obs[mask == 0] = -1

    model = CNFlowModel(d_x=d_x, d_y=d_y, k=k)
    loss = model.loss(x, y, t_obs)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.predict_outcome(x[:3], t[:3]).shape == (3, d_y)
    probs = model.predict_treatment_proba(x[:3], y[:3])
    assert probs.shape == (3, k)
    assert torch.allclose(probs.sum(-1), torch.ones(3), atol=1e-5)

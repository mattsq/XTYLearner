from xtylearner.data import load_synthetic_dataset
from xtylearner.models import TabJEPA


def test_tab_jepa_loss_and_forward():
    ds = load_synthetic_dataset(n_samples=10, d_x=2, seed=0)
    X, Y, T = ds.tensors
    model = TabJEPA(d_x=2, d_y=1, k=2)
    loss = model.loss(X, Y, T)
    assert loss.dim() == 0
    out = model.predict_outcome(X, T)
    assert out.shape == (10, 1)

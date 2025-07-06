import torch
from torch.utils.data import DataLoader

from xtylearner.data import load_mixed_synthetic_dataset
from xtylearner.models import MaskedTabularTransformer
from xtylearner.training import SupervisedTrainer


def test_masked_tabular_transformer_runs():
    ds = load_mixed_synthetic_dataset(n_samples=20, d_x=2, seed=0, label_ratio=0.7)
    X, Y, _ = ds.tensors
    model = MaskedTabularTransformer(d_x=2, y_bins=8, d_model=16, num_layers=2)
    model.set_y_range(float(Y.min()), float(Y.max()))

    loader = DataLoader(ds, batch_size=5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = SupervisedTrainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)

    x_row = X[0]
    y0 = model.predict_y(x_row, t_prompt=0, n_samples=2)
    y1 = model.predict_y(x_row, t_prompt=1, n_samples=2)
    assert isinstance(y0, float) and isinstance(y1, float)

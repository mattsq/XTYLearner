import torch
from torch.utils.data import DataLoader

from xtylearner.data import load_mixed_synthetic_dataset
from xtylearner.models import get_model
from xtylearner.training import Trainer


def test_em_trainer_runs():
    dataset = load_mixed_synthetic_dataset(n_samples=20, d_x=2, seed=0, label_ratio=0.5)
    loader = DataLoader(dataset, batch_size=10)
    model = get_model("em", k=2, max_iter=2)
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)
    metrics = trainer.evaluate(loader)
    assert set(metrics) >= {"loss", "treatment accuracy", "outcome rmse"}

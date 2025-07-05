import torch
from torch.utils.data import DataLoader
from xtylearner.data import load_toy_dataset
from xtylearner.models import CycleDual
from xtylearner.training import SupervisedTrainer


def test_supervised_trainer_runs():
    dataset = load_toy_dataset(n_samples=20, d_x=2, seed=0)
    loader = DataLoader(dataset, batch_size=5)
    model = CycleDual(d_x=2, d_y=1, k=2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = SupervisedTrainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)

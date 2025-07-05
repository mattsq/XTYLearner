import torch
from torch.utils.data import DataLoader
from xtylearner.data import load_toy_dataset
from xtylearner.models import CycleDual, MixtureOfFlows, MultiTask
from xtylearner.training import (
    SupervisedTrainer,
    M2VAE,
    SS_CEVAE,
    M2VAETrainer,
    CEVAETrainer,
)


def test_supervised_trainer_runs():
    dataset = load_toy_dataset(n_samples=20, d_x=2, seed=0)
    loader = DataLoader(dataset, batch_size=5)
    model = CycleDual(d_x=2, d_y=1, k=2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = SupervisedTrainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_flow_model_runs():
    dataset = load_toy_dataset(n_samples=20, d_x=2, seed=1)
    loader = DataLoader(dataset, batch_size=5)
    model = MixtureOfFlows(dim_x=2, dim_y=1, n_treat=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = SupervisedTrainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_multitask_model_runs():
    dataset = load_toy_dataset(n_samples=20, d_x=2, seed=2)
    loader = DataLoader(dataset, batch_size=5)
    model = MultiTask(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = SupervisedTrainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_m2vae_trainer_runs():
    dataset = load_toy_dataset(n_samples=20, d_x=2, seed=3)
    loader = DataLoader(dataset, batch_size=5)
    model = M2VAE(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = M2VAETrainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_cevae_trainer_runs():
    dataset = load_toy_dataset(n_samples=20, d_x=2, seed=4)
    loader = DataLoader(dataset, batch_size=5)
    model = SS_CEVAE(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = CEVAETrainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)

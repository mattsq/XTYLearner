import torch
from torch.utils.data import DataLoader
from xtylearner.data import load_toy_dataset, load_mixed_synthetic_dataset
from xtylearner.models import CycleDual, MixtureOfFlows, MultiTask
from xtylearner.training import SupervisedTrainer, GenerativeTrainer, DiffusionTrainer
from xtylearner.models import M2VAE, SS_CEVAE, JSBF, DiffusionCEVAE
from xtylearner.models import BridgeDiff, LTFlowDiff
from xtylearner.models import EnergyDiffusionImputer, JointEBM


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
    trainer = GenerativeTrainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_cevae_trainer_runs():
    dataset = load_toy_dataset(n_samples=20, d_x=2, seed=4)
    loader = DataLoader(dataset, batch_size=5)
    model = SS_CEVAE(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = GenerativeTrainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_supervised_trainer_mixed_dataset():
    dataset = load_mixed_synthetic_dataset(n_samples=20, d_x=2, seed=5, label_ratio=0.5)
    loader = DataLoader(dataset, batch_size=5)
    model = CycleDual(d_x=2, d_y=1, k=2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = SupervisedTrainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_cevae_trainer_mixed_dataset():
    dataset = load_mixed_synthetic_dataset(n_samples=20, d_x=2, seed=6, label_ratio=0.5)
    loader = DataLoader(dataset, batch_size=5)
    model = SS_CEVAE(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = GenerativeTrainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_jsbf_trainer_runs():
    dataset = load_mixed_synthetic_dataset(n_samples=20, d_x=2, seed=7, label_ratio=0.5)
    loader = DataLoader(dataset, batch_size=5)
    model = JSBF(d_x=2, d_y=1)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = DiffusionTrainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_diffusion_cevae_trainer_runs():
    dataset = load_mixed_synthetic_dataset(n_samples=20, d_x=2, seed=8, label_ratio=0.5)
    loader = DataLoader(dataset, batch_size=5)
    model = DiffusionCEVAE(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = GenerativeTrainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_bridge_diff_trainer_runs():
    dataset = load_mixed_synthetic_dataset(n_samples=20, d_x=2, seed=9, label_ratio=0.5)
    loader = DataLoader(dataset, batch_size=5)
    model = BridgeDiff(d_x=2, d_y=1)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = DiffusionTrainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_lt_flow_diff_trainer_runs():
    dataset = load_mixed_synthetic_dataset(
        n_samples=20, d_x=2, seed=10, label_ratio=0.5
    )
    loader = DataLoader(dataset, batch_size=5)
    model = LTFlowDiff(d_x=2, d_y=1)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = GenerativeTrainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_eg_ddi_trainer_runs():
    dataset = load_mixed_synthetic_dataset(
        n_samples=20, d_x=2, seed=11, label_ratio=0.5
    )
    loader = DataLoader(dataset, batch_size=5)
    model = EnergyDiffusionImputer(d_x=2, d_y=1)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = DiffusionTrainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_joint_ebm_trainer_runs():
    dataset = load_mixed_synthetic_dataset(
        n_samples=20, d_x=2, seed=12, label_ratio=0.5
    )
    loader = DataLoader(dataset, batch_size=5)
    model = JointEBM(d_x=2, d_y=1)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = SupervisedTrainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_predict_treatment_proba():
    dataset = load_toy_dataset(n_samples=6, d_x=2, seed=12)
    loader = DataLoader(dataset, batch_size=6)
    model = MultiTask(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = SupervisedTrainer(model, opt, loader)
    trainer.fit(1)
    X, Y, _ = next(iter(loader))
    probs = trainer.predict_treatment_proba(X, Y)
    assert probs.shape == (6, 2)
    assert torch.allclose(probs.sum(-1), torch.ones(6), atol=1e-5)

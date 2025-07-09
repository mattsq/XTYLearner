import torch
import pytest
from torch.utils.data import DataLoader
from xtylearner.data import (
    load_toy_dataset,
    load_mixed_synthetic_dataset,
    load_tabular_dataset,
)
import pandas as pd
import numpy as np
from xtylearner.models import CycleDual, MixtureOfFlows, MultiTask, DragonNet
from xtylearner.training import Trainer
from xtylearner.models import M2VAE, SS_CEVAE, JSBF, DiffusionCEVAE
from xtylearner.models import BridgeDiff, LTFlowDiff
from xtylearner.models import EnergyDiffusionImputer, JointEBM, GFlowNetTreatment
from xtylearner.models import GANITE
from xtylearner.models import ProbCircuitModel, LP_KNN


def test_supervised_trainer_runs():
    dataset = load_toy_dataset(n_samples=20, d_x=2, seed=0)
    loader = DataLoader(dataset, batch_size=5)
    model = CycleDual(d_x=2, d_y=1, k=2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_flow_model_runs():
    dataset = load_toy_dataset(n_samples=20, d_x=2, seed=1)
    loader = DataLoader(dataset, batch_size=5)
    model = MixtureOfFlows(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_multitask_model_runs():
    dataset = load_toy_dataset(n_samples=20, d_x=2, seed=2)
    loader = DataLoader(dataset, batch_size=5)
    model = MultiTask(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_multitask_handles_missing_labels():
    dataset = load_mixed_synthetic_dataset(
        n_samples=20, d_x=2, seed=42, label_ratio=0.5
    )
    loader = DataLoader(dataset, batch_size=5)
    model = MultiTask(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_dragon_net_runs():
    dataset = load_toy_dataset(n_samples=20, d_x=2, seed=18)
    loader = DataLoader(dataset, batch_size=5)
    model = DragonNet(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_dragon_net_multi_outcome():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=10).astype(np.float32),
            "x2": rng.normal(size=10).astype(np.float32),
            "y1": rng.normal(size=10).astype(np.float32),
            "y2": rng.normal(size=10).astype(np.float32),
            "t": rng.integers(0, 2, size=10).astype(np.int64),
        }
    )
    dataset = load_tabular_dataset(df, outcome_col=["y1", "y2"], treatment_col="t")
    loader = DataLoader(dataset, batch_size=5)
    model = DragonNet(d_x=2, d_y=2, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_m2vae_trainer_runs():
    dataset = load_toy_dataset(n_samples=20, d_x=2, seed=3)
    loader = DataLoader(dataset, batch_size=5)
    model = M2VAE(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_cevae_trainer_runs():
    dataset = load_toy_dataset(n_samples=20, d_x=2, seed=4)
    loader = DataLoader(dataset, batch_size=5)
    model = SS_CEVAE(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_supervised_trainer_mixed_dataset():
    dataset = load_mixed_synthetic_dataset(n_samples=20, d_x=2, seed=5, label_ratio=0.5)
    loader = DataLoader(dataset, batch_size=5)
    model = CycleDual(d_x=2, d_y=1, k=2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_cevae_trainer_mixed_dataset():
    dataset = load_mixed_synthetic_dataset(n_samples=20, d_x=2, seed=6, label_ratio=0.5)
    loader = DataLoader(dataset, batch_size=5)
    model = SS_CEVAE(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_jsbf_trainer_runs():
    dataset = load_mixed_synthetic_dataset(n_samples=20, d_x=2, seed=7, label_ratio=0.5)
    loader = DataLoader(dataset, batch_size=5)
    model = JSBF(d_x=2, d_y=1)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_diffusion_cevae_trainer_runs():
    dataset = load_mixed_synthetic_dataset(n_samples=20, d_x=2, seed=8, label_ratio=0.5)
    loader = DataLoader(dataset, batch_size=5)
    model = DiffusionCEVAE(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_bridge_diff_trainer_runs():
    dataset = load_mixed_synthetic_dataset(n_samples=20, d_x=2, seed=9, label_ratio=0.5)
    loader = DataLoader(dataset, batch_size=5)
    model = BridgeDiff(d_x=2, d_y=1, embed_dim=16)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, opt, loader)
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
    trainer = Trainer(model, opt, loader)
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
    trainer = Trainer(model, opt, loader)
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
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_gflownet_treatment_trainer_runs():
    dataset = load_mixed_synthetic_dataset(
        n_samples=20, d_x=2, seed=13, label_ratio=0.5
    )
    loader = DataLoader(dataset, batch_size=5)
    model = GFlowNetTreatment(d_x=2, d_y=1)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_ganite_trainer_runs():
    dataset = load_mixed_synthetic_dataset(
        n_samples=20, d_x=2, seed=17, label_ratio=0.5
    )
    loader = DataLoader(dataset, batch_size=5)
    model = GANITE(d_x=2, d_y=1)
    opt_g = torch.optim.Adam(model.parameters(), lr=0.001)
    opt_d = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, (opt_g, opt_d), loader)
    trainer.fit(1)
    loss = trainer.evaluate(loader)
    assert isinstance(loss, float)


def test_predict_treatment_proba():
    dataset = load_toy_dataset(n_samples=6, d_x=2, seed=12)
    loader = DataLoader(dataset, batch_size=6)
    model = MultiTask(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)
    X, Y, _ = next(iter(loader))
    probs = trainer.predict_treatment_proba(X, Y)
    assert probs.shape == (6, 2)
    assert torch.allclose(probs.sum(-1), torch.ones(6), atol=1e-5)


def test_trainer_with_scheduler():
    dataset = load_toy_dataset(n_samples=10, d_x=2, seed=14)
    loader = DataLoader(dataset, batch_size=5)
    model = CycleDual(d_x=2, d_y=1, k=2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.1)
    trainer = Trainer(model, opt, loader, scheduler=sched)
    trainer.fit(2)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.001, rel=1e-6)


def test_trainer_handles_model_without_to():
    dataset = load_toy_dataset(n_samples=2, d_x=2, seed=15)
    loader = DataLoader(dataset, batch_size=1)
    model = ProbCircuitModel(min_instances_slice=5)
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
    trainer = Trainer(model, opt, loader)
    assert trainer.model is model


def test_labelprop_trainer_runs():
    dataset = load_toy_dataset(n_samples=20, d_x=2, seed=16)
    loader = DataLoader(dataset, batch_size=5)
    model = LP_KNN(n_neighbors=3)
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
    trainer = Trainer(model, opt, loader)
    trainer.fit(1)
    acc = trainer.evaluate(loader)
    assert isinstance(acc, float)

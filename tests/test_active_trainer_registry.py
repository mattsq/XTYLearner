import torch
from torch.utils.data import DataLoader, TensorDataset

from xtylearner.data import load_mixed_synthetic_dataset
from xtylearner.models import get_model, get_model_names
from xtylearner.models.ss_dml import _HAS_DOUBLEML
from xtylearner.training import ActiveTrainer
from xtylearner.active import QueryStrategy


class DummyStrategy(QueryStrategy):
    def forward(self, model, X_unlab, rep_fn, batch_size):
        return torch.zeros(len(X_unlab))


def _make_optimizer(model: torch.nn.Module):
    params = [p for p in getattr(model, "parameters", lambda: [])() if p.requires_grad]
    if not params:
        params = [torch.zeros(1, requires_grad=True)]
    return torch.optim.SGD(params, lr=0.01)


def _make_gan_optimizer(model: torch.nn.Module):
    opt_g = torch.optim.SGD(model.parameters(), lr=0.01)
    opt_d = torch.optim.SGD(model.parameters(), lr=0.01)
    return opt_g, opt_d


def test_active_trainer_runs_for_all_models():
    ds = load_mixed_synthetic_dataset(n_samples=20, d_x=2, seed=0, label_ratio=0.5)
    X, Y, T = ds.tensors
    base_loader = DataLoader(ds, batch_size=4)

    for name in get_model_names():
        if name == "ss_dml" and not _HAS_DOUBLEML:
            continue

        kwargs = {"d_x": 2, "d_y": 1, "k": 2}
        if name == "lp_knn":
            kwargs["n_neighbors"] = 3
        if name == "ctm_t":
            kwargs = {"d_in": 4}

        model = get_model(name, **kwargs)

        if name == "deconfounder_cfm":
            t = torch.nn.functional.one_hot(T.clamp_min(0), 2).float()
            t[T < 0] = float("nan")
            loader = DataLoader(TensorDataset(X, Y, t), batch_size=4)
        else:
            loader = base_loader

        if hasattr(model, "loss_G") and hasattr(model, "loss_D"):
            opt = _make_gan_optimizer(model)
        else:
            opt = _make_optimizer(model)

        trainer = ActiveTrainer(model, opt, loader, DummyStrategy(), budget=2, batch=1)
        trainer.fit(1)



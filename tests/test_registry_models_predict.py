import torch
from torch.utils.data import DataLoader

from xtylearner.data import load_toy_dataset
from xtylearner.models import get_model, get_model_names
from xtylearner.models.ss_dml import _HAS_DOUBLEML
from xtylearner.training import Trainer


def _make_optimizer(model: torch.nn.Module):
    params = []
    if hasattr(model, "parameters"):
        params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        params = [torch.zeros(1, requires_grad=True)]
    return torch.optim.Adam(params, lr=0.01)


def _make_gan_optimizer(model: torch.nn.Module):
    opt_g = torch.optim.Adam(model.parameters(), lr=0.01)
    opt_d = torch.optim.Adam(model.parameters(), lr=0.01)
    return opt_g, opt_d


def _run_predict(trainer: Trainer, x: torch.Tensor, t: torch.Tensor):
    if t.dim() > 1:
        t_scalar = int(t[0].argmax().item())
    else:
        t_scalar = int(t[0].item())
    for args in [(x, t), (x, t_scalar), (x,), (len(x),)]:
        try:
            return trainer.predict(*args)
        except Exception:
            continue
    raise RuntimeError("predict not supported")


def _first(probs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]):
    if isinstance(probs, (list, tuple)):
        return probs[0]
    return probs


def test_all_registered_models_predict_and_proba():
    ds = load_toy_dataset(n_samples=8, d_x=2, seed=0)
    base_loader = DataLoader(ds, batch_size=4)
    X, Y, T = next(iter(base_loader))

    for name in get_model_names():
        if name == "ss_dml" and not _HAS_DOUBLEML:
            continue
        kwargs = {"d_x": 2, "d_y": 1, "k": 2}
        if name == "lp_knn":
            kwargs["n_neighbors"] = 3
        if name == "ctm_t":
            kwargs = {"d_in": 4}
        model = get_model(name, **kwargs)
        if hasattr(model, "loss_G") and hasattr(model, "loss_D"):
            opt = _make_gan_optimizer(model)
        else:
            opt = _make_optimizer(model)

        loader = base_loader
        t_pred = T

        trainer = Trainer(model, opt, loader)
        trainer.fit(1)
        preds = _run_predict(trainer, X, t_pred)
        if hasattr(preds, "shape"):
            assert preds.shape[0] == X.shape[0]
        else:
            assert len(preds) == X.shape[0]

        probs = trainer.predict_treatment_proba(X, Y)
        probs = _first(probs)
        assert probs.shape[0] == X.shape[0]
        ones = torch.ones(len(X), dtype=probs.dtype)
        assert torch.allclose(probs.sum(-1), ones, atol=1e-5)

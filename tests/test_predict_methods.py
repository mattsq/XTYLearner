import torch
from xtylearner.data import load_toy_dataset
from xtylearner.models import get_model, get_model_names


def _run_predict_outcome(model, x, t, t_onehot):
    scalar = int(t[0].item())
    attempts = [
        (x, t),
        (x, t_onehot),
        (x, scalar),
        (x.numpy(), t.numpy()),
        (x.numpy(), t_onehot.numpy()),
        (x.numpy(), scalar),
    ]
    for args in attempts:
        try:
            return model.predict_outcome(*args)
        except Exception:
            continue
    raise RuntimeError("predict_outcome not supported")


def test_all_models_have_predict_methods():
    ds = load_toy_dataset(n_samples=6, d_x=2, seed=42)
    X, Y, T = ds.tensors
    T_oh = torch.nn.functional.one_hot(T, 2).float()
    for name in get_model_names():
        model = get_model(name, d_x=2, d_y=1, k=2)
        out = _run_predict_outcome(model, X, T, T_oh)
        if isinstance(out, torch.Tensor):
            assert out.shape[0] == X.shape[0]
        else:
            assert len(out) == X.shape[0]
        attempts = [
            (X, Y),
            (torch.cat([X, Y], dim=1),),
            (X.numpy(), Y.numpy()),
            (torch.cat([X, Y], dim=1).numpy(),),
        ]
        for args in attempts:
            try:
                probs = model.predict_treatment_proba(*args)
                break
            except Exception:
                continue
        if isinstance(probs, (list, tuple)):
            probs = probs[0]
        if not isinstance(probs, torch.Tensor):
            probs = torch.as_tensor(probs)
        assert probs.shape[0] == X.shape[0]

import torch
from xtylearner.data import load_mixed_synthetic_dataset
from xtylearner.models import MeanTeacher


def test_mean_teacher_continuous():
    """Test MeanTeacher with continuous treatments (k=None)."""
    ds = load_mixed_synthetic_dataset(
        n_samples=10, d_x=2, seed=0, label_ratio=0.5, continuous_treatment=True
    )
    X, Y, T_obs = ds.tensors
    model = MeanTeacher(d_x=2, d_y=1, k=None)
    loss = model.loss(X, Y, T_obs)
    assert loss.dim() == 0
    out = model.predict_outcome(X, torch.zeros(len(X)))
    assert out.shape == (len(X), 1)


def test_mean_teacher_discrete():
    """Test MeanTeacher with discrete treatments (k=2)."""
    ds = load_mixed_synthetic_dataset(
        n_samples=10, d_x=2, seed=0, label_ratio=0.5, continuous_treatment=False
    )
    X, Y, T_obs = ds.tensors
    model = MeanTeacher(d_x=2, d_y=1, k=2)
    loss = model.loss(X, Y, T_obs)
    assert loss.dim() == 0
    out = model.predict_outcome(X, torch.zeros(len(X), dtype=torch.long))
    assert out.shape == (len(X), 1)

    # Test treatment probability prediction
    probs = model.predict_treatment_proba(X, Y)
    assert probs.shape == (len(X), 2)
    assert torch.allclose(probs.sum(-1), torch.ones(len(X)), atol=1e-5)


def test_mean_teacher_continuous_predict_treatment():
    """Test treatment prediction for continuous treatments."""
    ds = load_mixed_synthetic_dataset(
        n_samples=10, d_x=2, seed=1, label_ratio=0.5, continuous_treatment=True
    )
    X, Y, T_obs = ds.tensors
    model = MeanTeacher(d_x=2, d_y=1, k=None)

    # Train for one step
    loss = model.loss(X, Y, T_obs)
    loss.backward()
    model.step()

    # Test treatment prediction (should return scalar values, not probabilities)
    t_pred = model.predict_treatment_proba(X, Y)
    assert t_pred.shape == (len(X),)
    assert t_pred.dtype == torch.float32


def test_mean_teacher_continuous_ood_score():
    """Test OOD score returns zeros for continuous treatments."""
    ds = load_mixed_synthetic_dataset(
        n_samples=10, d_x=2, seed=2, label_ratio=0.5, continuous_treatment=True
    )
    X, Y, T_obs = ds.tensors
    model = MeanTeacher(d_x=2, d_y=1, k=None)

    # OOD score should return zeros for continuous treatments
    ood_scores = model.predict_ood_score(X, Y)
    assert ood_scores.shape == (len(X),)
    assert torch.allclose(ood_scores, torch.zeros(len(X)))

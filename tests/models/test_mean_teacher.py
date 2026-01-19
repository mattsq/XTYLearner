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
    """Test OOD score returns zeros for continuous treatments without variance."""
    ds = load_mixed_synthetic_dataset(
        n_samples=10, d_x=2, seed=2, label_ratio=0.5, continuous_treatment=True
    )
    X, Y, T_obs = ds.tensors
    model = MeanTeacher(d_x=2, d_y=1, k=None, ood_weighting=False)

    # OOD score should return zeros for continuous treatments without variance
    ood_scores = model.predict_ood_score(X, Y)
    assert ood_scores.shape == (len(X),)
    assert torch.allclose(ood_scores, torch.zeros(len(X)))


def test_mean_teacher_continuous_with_variance():
    """Test MeanTeacher with continuous treatments and variance-based OOD detection."""
    ds = load_mixed_synthetic_dataset(
        n_samples=10, d_x=2, seed=3, label_ratio=0.5, continuous_treatment=True
    )
    X, Y, T_obs = ds.tensors

    # Create model with OOD weighting enabled for continuous treatments
    model = MeanTeacher(d_x=2, d_y=1, k=None, ood_weighting=True)

    # Verify variance heads are enabled
    assert model.use_variance_head is True

    # Check that consistency head outputs 2 values (mean + log_var)
    assert model.student_cons_head.out_features == 2
    assert model.teacher_head.out_features == 2

    # Test loss computation works
    loss = model.loss(X, Y, T_obs)
    assert loss.dim() == 0
    assert not torch.isnan(loss)

    # Test backward pass
    loss.backward()
    model.step()

    # Test outcome prediction still works
    out = model.predict_outcome(X, torch.zeros(len(X)))
    assert out.shape == (len(X), 1)


def test_mean_teacher_continuous_variance_treatment_prediction():
    """Test treatment prediction returns mean when using variance heads."""
    ds = load_mixed_synthetic_dataset(
        n_samples=10, d_x=2, seed=4, label_ratio=0.5, continuous_treatment=True
    )
    X, Y, T_obs = ds.tensors

    # Model with variance heads
    model = MeanTeacher(d_x=2, d_y=1, k=None, ood_weighting=True)

    # Train for one step
    loss = model.loss(X, Y, T_obs)
    loss.backward()
    model.step()

    # Test treatment prediction (should return mean predictions)
    t_pred = model.predict_treatment_proba(X, Y)
    assert t_pred.shape == (len(X),)
    assert t_pred.dtype == torch.float32


def test_mean_teacher_continuous_variance_ood_scores():
    """Test OOD scores are computed from variance for continuous treatments."""
    ds = load_mixed_synthetic_dataset(
        n_samples=10, d_x=2, seed=5, label_ratio=0.5, continuous_treatment=True
    )
    X, Y, T_obs = ds.tensors

    # Model with variance heads
    model = MeanTeacher(d_x=2, d_y=1, k=None, ood_weighting=True)

    # Train for a few steps to get meaningful variance predictions
    for _ in range(5):
        loss = model.loss(X, Y, T_obs)
        loss.backward()
        model.step()

    # OOD scores should NOT be all zeros with variance heads
    ood_scores = model.predict_ood_score(X, Y)
    assert ood_scores.shape == (len(X),)
    assert not torch.allclose(ood_scores, torch.zeros(len(X)))

    # OOD scores should be in [0, 1] range
    assert (ood_scores >= 0).all()
    assert (ood_scores <= 1).all()


def test_mean_teacher_continuous_backward_compatibility():
    """Test that ood_weighting=False maintains original behavior for continuous."""
    ds = load_mixed_synthetic_dataset(
        n_samples=10, d_x=2, seed=6, label_ratio=0.5, continuous_treatment=True
    )
    X, Y, T_obs = ds.tensors

    # Model without variance heads (backward compatible)
    model = MeanTeacher(d_x=2, d_y=1, k=None, ood_weighting=False)

    # Verify variance heads are NOT enabled
    assert model.use_variance_head is False

    # Check that consistency head outputs 1 value (original behavior)
    assert model.student_cons_head.out_features == 1
    assert model.teacher_head.out_features == 1

    # Test loss computation works
    loss = model.loss(X, Y, T_obs)
    assert loss.dim() == 0
    assert not torch.isnan(loss)

    # Test backward pass
    loss.backward()
    model.step()

    # OOD scores should be zeros (original behavior)
    ood_scores = model.predict_ood_score(X, Y)
    assert torch.allclose(ood_scores, torch.zeros(len(X)))

"""Tests for DAG-aware Transformer model."""

import torch
import pytest
from xtylearner.data import load_synthetic_dataset
from xtylearner.models import DAGTransformer
from xtylearner.models.dag_transformer import DAGAwareAttention, DAGAwareTransformerLayer


def test_dag_transformer_basic():
    """Test basic initialization and forward pass."""
    model = DAGTransformer(d_x=3, d_y=1, k=2)
    x = torch.randn(5, 3)
    t = torch.tensor([0, 1, 0, 1, 0])
    y_pred = model.forward(x, t)
    assert y_pred.shape == (5, 1)


def test_dag_transformer_loss():
    """Test loss computation with observed and missing treatments."""
    ds = load_synthetic_dataset(n_samples=20, d_x=4, seed=42)
    X, Y, T = ds.tensors

    model = DAGTransformer(d_x=4, d_y=1, k=2)
    loss = model.loss(X, Y, T)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_dag_transformer_semi_supervised():
    """Test semi-supervised learning with missing treatments."""
    X = torch.randn(10, 3)
    Y = torch.randn(10)
    # Some treatments are observed, some are missing (-1)
    T = torch.tensor([0, 1, -1, 0, -1, 1, -1, 0, 1, -1])

    model = DAGTransformer(d_x=3, d_y=1, k=2)
    loss = model.loss(X, Y, T)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_dag_transformer_predict_y():
    """Test predict_y for single sample."""
    model = DAGTransformer(d_x=3, d_y=1, k=2)
    x = torch.randn(3)
    y0 = model.predict_y(x, t_prompt=0)
    y1 = model.predict_y(x, t_prompt=1)
    assert isinstance(y0, float)
    assert isinstance(y1, float)


def test_dag_transformer_default_dag():
    """Test that default DAG has correct structure."""
    model = DAGTransformer(d_x=3, d_y=1, k=2)
    dag_mask = model.get_dag_mask()

    # DAG shape should be (num_vars, num_vars) = (5, 5)
    # 3 covariates + 1 treatment + 1 outcome
    assert dag_mask.shape == (5, 5)

    # Covariates (0,1,2) should influence treatment (3)
    assert dag_mask[0, 3] == 1.0
    assert dag_mask[1, 3] == 1.0
    assert dag_mask[2, 3] == 1.0

    # Covariates should influence outcome (4)
    assert dag_mask[0, 4] == 1.0
    assert dag_mask[1, 4] == 1.0
    assert dag_mask[2, 4] == 1.0

    # Treatment (3) should influence outcome (4)
    assert dag_mask[3, 4] == 1.0

    # Diagonal should be 1 (self-attention)
    assert torch.all(torch.diag(dag_mask) == 1.0)


def test_dag_transformer_custom_dag():
    """Test with custom DAG prior."""
    # Create custom DAG: only first covariate affects treatment
    custom_dag = torch.zeros(5, 5)
    custom_dag[0, 3] = 1.0  # X_0 -> T
    custom_dag[0, 4] = 1.0  # X_0 -> Y
    custom_dag[1, 4] = 1.0  # X_1 -> Y
    custom_dag[3, 4] = 1.0  # T -> Y
    custom_dag.fill_diagonal_(1.0)  # Self-attention

    model = DAGTransformer(d_x=3, d_y=1, k=2, dag_prior=custom_dag)
    dag_mask = model.get_dag_mask()

    assert torch.allclose(dag_mask, custom_dag)


def test_dag_transformer_learn_dag():
    """Test DAG structure learning."""
    model = DAGTransformer(d_x=3, d_y=1, k=2, learn_dag=True)

    # DAG should be learnable
    assert hasattr(model, 'dag_logits')
    assert model.dag_logits.requires_grad

    # Get DAG mask
    dag_mask = model.get_dag_mask()
    assert dag_mask.shape == (5, 5)

    # Acyclicity loss should be computed
    acyc_loss = model.dag_acyclicity_loss()
    assert acyc_loss.dim() == 0


def test_dag_transformer_learn_dag_with_prior():
    """Test DAG learning with prior structure."""
    prior = torch.zeros(5, 5)
    prior[0, 3] = 0.8  # Strong prior: X_0 -> T
    prior[3, 4] = 0.9  # Strong prior: T -> Y
    prior.fill_diagonal_(1.0)

    model = DAGTransformer(d_x=3, d_y=1, k=2, learn_dag=True, dag_prior=prior)

    # Logits should be initialized from prior
    assert model.dag_logits.requires_grad


def test_dag_transformer_ipw_estimator():
    """Test inverse propensity weighting estimator."""
    model = DAGTransformer(d_x=3, d_y=1, k=2, estimator='ipw')

    # Should have propensity head
    assert hasattr(model, 'propensity_head')

    # Test propensity prediction
    x = torch.randn(5, 3)
    props = model.predict_propensity(x)
    assert props.shape == (5, 2)
    assert torch.allclose(props.sum(dim=1), torch.ones(5), atol=1e-5)


def test_dag_transformer_aipw_estimator():
    """Test doubly robust (AIPW) estimator."""
    model = DAGTransformer(d_x=3, d_y=1, k=2, estimator='aipw')

    # Should have both outcome and propensity heads
    assert hasattr(model, 'outcome_head')
    assert hasattr(model, 'propensity_head')
    assert hasattr(model, 'alpha')

    # Test forward pass
    x = torch.randn(5, 3)
    t = torch.tensor([0, 1, 0, 1, 0])
    y_pred = model.forward(x, t)
    assert y_pred.shape == (5, 1)

    # Test propensity
    props = model.predict_propensity(x)
    assert props.shape == (5, 2)


def test_dag_aware_attention():
    """Test DAG-aware attention mechanism."""
    attn = DAGAwareAttention(d_model=64, nhead=4, dropout=0.1)

    x = torch.randn(2, 5, 64)  # (batch, seq_len, d_model)

    # Test without DAG mask
    out = attn(x)
    assert out.shape == (2, 5, 64)

    # Test with DAG mask
    dag_mask = torch.ones(5, 5)
    dag_mask[0, 1:] = 0  # First variable can't attend to others
    out = attn(x, dag_mask=dag_mask)
    assert out.shape == (2, 5, 64)


def test_dag_aware_transformer_layer():
    """Test single DAG-aware transformer layer."""
    layer = DAGAwareTransformerLayer(d_model=64, nhead=4, dim_feedforward=128)

    x = torch.randn(2, 5, 64)
    dag_mask = torch.ones(5, 5)

    out = layer(x, dag_mask=dag_mask)
    assert out.shape == (2, 5, 64)


def test_dag_transformer_embed_sequence():
    """Test sequence embedding."""
    model = DAGTransformer(d_x=3, d_y=1, k=2)

    x = torch.randn(4, 3)
    t = torch.tensor([0, 1, 0, 1])
    y = torch.randn(4)

    # Test with all inputs
    tokens = model.embed_sequence(x, t, y)
    assert tokens.shape == (4, 5, model.d_model)

    # Test with missing outcome
    tokens = model.embed_sequence(x, t, y=None)
    assert tokens.shape == (4, 5, model.d_model)

    # Test with missing treatment
    t_missing = torch.tensor([0, -1, 1, -1])
    tokens = model.embed_sequence(x, t_missing, y)
    assert tokens.shape == (4, 5, model.d_model)


def test_dag_transformer_different_params():
    """Test model with different hyperparameters."""
    model = DAGTransformer(
        d_x=5,
        d_y=1,
        k=3,
        d_model=256,
        num_layers=6,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.2,
    )

    x = torch.randn(3, 5)
    t = torch.tensor([0, 1, 2])

    y_pred = model.forward(x, t)
    assert y_pred.shape == (3, 1)


def test_dag_transformer_invalid_estimator():
    """Test that invalid estimator raises error."""
    with pytest.raises(ValueError, match="Unknown estimator"):
        DAGTransformer(d_x=3, d_y=1, k=2, estimator='invalid')


def test_dag_transformer_training_mode():
    """Test model behavior in training vs eval mode."""
    model = DAGTransformer(d_x=3, d_y=1, k=2, learn_dag=True)

    x = torch.randn(5, 3)
    t = torch.tensor([0, 1, 0, 1, 0])

    # Training mode
    model.train()
    dag_mask_train = model.get_dag_mask()

    # Eval mode
    model.eval()
    dag_mask_eval = model.get_dag_mask()

    # In eval mode, DAG should be thresholded (binary)
    assert torch.all((dag_mask_eval == 0) | (dag_mask_eval == 1))


def test_dag_transformer_gradient_flow():
    """Test that gradients flow through the model."""
    model = DAGTransformer(d_x=3, d_y=1, k=2)

    x = torch.randn(5, 3, requires_grad=True)
    y = torch.randn(5)
    t = torch.tensor([0, 1, 0, 1, 0])

    loss = model.loss(x, y, t)
    loss.backward()

    # Check gradients exist
    assert x.grad is not None
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_dag_transformer_acyclicity_loss():
    """Test that acyclicity loss prevents cycles."""
    model = DAGTransformer(d_x=3, d_y=1, k=2, learn_dag=True)

    acyc_loss = model.dag_acyclicity_loss()
    assert acyc_loss.dim() == 0
    assert acyc_loss.item() >= 0

    # Non-learnable DAG should have zero acyclicity loss
    model_fixed = DAGTransformer(d_x=3, d_y=1, k=2, learn_dag=False)
    acyc_loss_fixed = model_fixed.dag_acyclicity_loss()
    assert acyc_loss_fixed.item() == 0.0

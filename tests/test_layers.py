import pytest
import torch
import torch.nn as nn

from xtylearner.models.layers import make_mlp, ColumnEmbedder, apply_column_mask


def test_make_mlp_default():
    mlp = make_mlp([2, 3, 1])
    assert len(mlp) == 3
    assert isinstance(mlp[0], nn.Linear)
    assert isinstance(mlp[1], nn.ReLU)
    assert isinstance(mlp[2], nn.Linear)


def test_make_mlp_with_norm_and_dropout():
    mlp = make_mlp(
        [2, 3, 1], activation=nn.Tanh, dropout=0.2, norm_layer=nn.BatchNorm1d
    )
    assert isinstance(mlp[0], nn.Linear)
    assert isinstance(mlp[1], nn.BatchNorm1d)
    assert isinstance(mlp[2], nn.Tanh)
    assert isinstance(mlp[3], nn.Dropout)
    assert isinstance(mlp[4], nn.Linear)


def test_make_mlp_dropout_sequence():
    mlp = make_mlp([2, 3, 4, 1], dropout=[0.1, 0.3])
    layers = list(mlp)
    assert isinstance(layers[2], nn.Dropout)
    assert isinstance(layers[5], nn.Dropout)


def test_make_mlp_dropout_mismatch():
    with pytest.raises(ValueError):
        make_mlp([2, 3, 1], dropout=[0.1, 0.2])


def test_column_embedder_and_mask():
    emb = ColumnEmbedder(d_x=2, d_y=1, k=2, d_embed=4)
    X = torch.randn(3, 2)
    T = torch.tensor([0, 1, 0])
    T_tok = torch.nn.functional.one_hot(T, 2).float()
    toks = emb(X, T_tok)
    assert toks.shape == (3, 3, 4)

    masked, mask = apply_column_mask(toks, 0.5)
    assert masked.shape == toks.shape
    assert mask.shape == (3, 3)

from typing import Callable, Sequence, Iterable, Optional

import torch
import torch.nn as nn


def make_mlp(
    dims: Sequence[int],
    activation: Callable[[], nn.Module] = nn.ReLU,
    *,
    dropout: Optional[Iterable[float]] | float | None = None,
    norm_layer: Optional[Callable[[int], nn.Module]] = None,
) -> nn.Sequential:
    """Construct a simple feedforward neural network.

    Parameters
    ----------
    dims : Sequence[int]
        List of layer dimensions ``[in_dim, hidden1, ..., out_dim]``.
    activation : Callable[[], nn.Module], optional
        Activation constructor inserted between linear layers. Defaults to ``nn.ReLU``.
    dropout : float | Iterable[float] | None, optional
        Dropout probability after activation for each hidden layer. If a single
        float is provided it is used for all hidden layers. ``None`` disables
        dropout.
    norm_layer : Callable[[int], nn.Module] | None, optional
        Normalisation layer constructor applied to each hidden layer. The integer
        argument corresponds to the size of the current layer.

    Returns
    -------
    nn.Sequential
        The assembled multi-layer perceptron.
    """

    n_layers = len(dims) - 1
    if isinstance(dropout, Iterable) and not isinstance(dropout, (str, bytes)):
        dropouts = list(dropout)
        if len(dropouts) != n_layers - 1:
            raise ValueError("dropout iterable must match number of hidden layers")
    elif dropout is None:
        dropouts = [None] * (n_layers - 1)
    else:
        dropouts = [float(dropout)] * (n_layers - 1)

    layers = []
    for i in range(n_layers):
        in_dim, out_dim = dims[i], dims[i + 1]
        layers.append(nn.Linear(in_dim, out_dim))
        if i < n_layers - 1:
            if norm_layer is not None:
                layers.append(norm_layer(out_dim))
            layers.append(activation())
            p = dropouts[i]
            if p is not None and p > 0:
                layers.append(nn.Dropout(p))

    return nn.Sequential(*layers)


class ColumnEmbedder(nn.Module):
    """Embed numeric columns and treatment tokens into a sequence."""

    def __init__(self, d_x, d_y, k, d_embed):
        super().__init__()
        self.proj_x = nn.Linear(d_x, d_embed)
        self.proj_y = nn.Linear(d_y, d_embed)
        if k is None:
            self.proj_t = nn.Linear(1, d_embed)
        else:
            self.proj_t = nn.Linear(k, d_embed)

    def forward(self, X, T_tok):
        B = X.size(0)
        y_placeholder = torch.zeros(
            B, self.proj_y.in_features, device=X.device, dtype=X.dtype
        )
        toks = torch.stack(
            [self.proj_x(X), self.proj_t(T_tok), self.proj_y(y_placeholder)],
            dim=1,
        )
        return toks


def apply_column_mask(tokens: torch.Tensor, ratio: float):
    """Mask random columns with the given ratio."""
    B, N, D = tokens.shape
    mask = torch.rand(B, N, device=tokens.device) < ratio
    masked_tokens = tokens.clone()
    masked_tokens[mask] = 0.0
    return masked_tokens, mask

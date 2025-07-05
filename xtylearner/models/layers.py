import torch.nn as nn
from typing import Sequence, Callable


def make_mlp(
    dims: Sequence[int], activation: Callable[[], nn.Module] = nn.ReLU
) -> nn.Sequential:
    """Construct a simple feedforward neural network.

    Parameters
    ----------
    dims : Sequence[int]
        List of layer dimensions ``[in_dim, hidden1, ..., out_dim]``.
    activation : Callable[[], nn.Module], optional
        Activation constructor inserted between linear layers. Defaults to ``nn.ReLU``.
    """
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)

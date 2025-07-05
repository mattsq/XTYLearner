"""Model architectures available in XTYLearner."""

from .cycle_dual import CycleDual
from .flow_ssc import MixtureOfFlows
from .multitask_selftrain import MultiTask
from .generative import M2VAE, SS_CEVAE
from .jsbf_model import JSBF
from .registry import get_model

__all__ = [
    "CycleDual",
    "MixtureOfFlows",
    "MultiTask",
    "M2VAE",
    "SS_CEVAE",
    "JSBF",
    "get_model",
]

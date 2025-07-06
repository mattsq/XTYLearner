"""Model architectures available in XTYLearner."""

from .cycle_dual import CycleDual
from .flow_ssc import MixtureOfFlows
from .multitask_selftrain import MultiTask
from .generative import M2VAE, SS_CEVAE, DiffusionCEVAE
from .energy_diffusion_imputer import EnergyDiffusionImputer
from .joint_ebm import JointEBM
from .jsbf_model import JSBF
from .bridge_diff import BridgeDiff
from .lt_flow_diff import LTFlowDiff
from .masked_tabular_transformer import MaskedTabularTransformer
from .registry import get_model

__all__ = [
    "CycleDual",
    "MixtureOfFlows",
    "MultiTask",
    "M2VAE",
    "SS_CEVAE",
    "DiffusionCEVAE",
    "JSBF",
    "BridgeDiff",
    "LTFlowDiff",
    "EnergyDiffusionImputer",
    "JointEBM",
    "MaskedTabularTransformer",
    "get_model",
]

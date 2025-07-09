"""Model architectures available in XTYLearner."""

from .cycle_dual import CycleDual
from .flow_ssc import MixtureOfFlows
from .multitask_selftrain import MultiTask
from .generative import M2VAE, SS_CEVAE, DiffusionCEVAE
from .cevae_m_model import CEVAE_M
from .energy_diffusion_imputer import EnergyDiffusionImputer
from .joint_ebm import JointEBM
from .jsbf_model import JSBF
from .bridge_diff import BridgeDiff
from .lt_flow_diff import LTFlowDiff
from .prob_circuit_model import ProbCircuitModel
from .masked_tabular_transformer import MaskedTabularTransformer
from .gflownet_treatment import GFlowNetTreatment
from .dragon_net import DragonNet
from .gnn_scm import GNN_SCM
from .em_model import EMModel
from .labelprop import LP_KNN
from .mean_teacher import MeanTeacher
from .vime import VIME
from .vat import VAT_Model
from .fixmatch import FixMatch
from .ss_dml import SSDMLModel
from .ganite import GANITE
from .deconfounder_model import DeconfounderCFM
from .registry import get_model, get_model_names, get_model_args

__all__ = [
    "CycleDual",
    "MixtureOfFlows",
    "MultiTask",
    "M2VAE",
    "SS_CEVAE",
    "DiffusionCEVAE",
    "CEVAE_M",
    "JSBF",
    "BridgeDiff",
    "LTFlowDiff",
    "EnergyDiffusionImputer",
    "JointEBM",
    "MaskedTabularTransformer",
    "DragonNet",
    "ProbCircuitModel",
    "GFlowNetTreatment",
    "EMModel",
    "LP_KNN",
    "MeanTeacher",
    "VIME",
    "VAT_Model",
    "FixMatch",
    "SSDMLModel",
    "GANITE",
    "DeconfounderCFM",
    "GNN_SCM",
    "get_model",
    "get_model_names",
    "get_model_args",
]

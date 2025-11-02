"""Active learning utilities and query strategies."""

from .strategies import (
    QueryStrategy,
    EntropyT,
    DeltaCATE,
    CATEUncertainty,
    ConformalCATEIntervalStrategy,
    DebiasedCoverageAcquisition,
    FCCMRadius,
    STRATEGIES,
)
from .calibration import ConformalCalibrator, build_conformal_calibrator
from .registry import get_strategy

__all__ = [
    "QueryStrategy",
    "EntropyT",
    "DeltaCATE",
    "CATEUncertainty",
    "ConformalCATEIntervalStrategy",
    "DebiasedCoverageAcquisition",
    "FCCMRadius",
    "STRATEGIES",
    "ConformalCalibrator",
    "build_conformal_calibrator",
    "get_strategy",
]

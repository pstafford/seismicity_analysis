"""
Core data types for seismicity analysis.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

LOG10 = np.log(10.0)


@dataclass
class Catalogue:
    """Earthquake catalogue with per-event magnitudes, uncertainties, and metadata."""
    ml: np.ndarray
    mw: np.ndarray
    sigma_ml: np.ndarray
    year: np.ndarray
    lat: np.ndarray
    lon: np.ndarray

    def __len__(self):
        return len(self.ml)


@dataclass
class CompletenessModel:
    """
    Completeness model defining magnitude-dependent start years of complete recording.
    Thresholds must be in descending order of magnitude.
    """
    mw_thresholds: np.ndarray
    start_years: np.ndarray
    end_year: float

    def __post_init__(self):
        self.mw_thresholds = np.asarray(self.mw_thresholds, dtype=float)
        self.start_years = np.asarray(self.start_years, dtype=float)
        if len(self.mw_thresholds) != len(self.start_years):
            raise ValueError("mw_thresholds and start_years must have same length")
        if not all(self.mw_thresholds[i] >= self.mw_thresholds[i + 1]
                    for i in range(len(self.mw_thresholds) - 1)):
            raise ValueError("mw_thresholds must be in descending order")


@dataclass
class CompletenessBin:
    """A magnitude-observation time bin derived from a completeness model."""
    mw_lo: float
    mw_hi: float
    t_obs: float


@dataclass
class GRConfig:
    """Configuration for Gutenberg-Richter magnitude-frequency analysis."""
    mw_min: float = 3.0
    mw_max: float = 6.5
    mw_floor: float = None
    dm_ml: float = 0.1
    sigma_ml_default: float = 0.25

    def __post_init__(self):
        if self.mw_floor is None:
            self.mw_floor = self.mw_min - 1.0
        if self.mw_min >= self.mw_max:
            raise ValueError("mw_min must be less than mw_max")


@dataclass
class GRResult:
    """Results from a Gutenberg-Richter parameter estimation."""
    b_mean: float
    b_sd: float
    b_q025: float
    b_q975: float
    lambda_mean: float
    lambda_sd: float
    lambda_q025: float
    lambda_q975: float
    rho_lb: float
    cov_matrix: np.ndarray
    method: str
    samples: Optional[dict]
    scatter_in_pct: float
    n_events: int

    @staticmethod
    def create(*, b_mean, b_sd, lambda_mean, lambda_sd, method, n_events,
               b_q025=None, b_q975=None, lambda_q025=None, lambda_q975=None,
               rho_lb=0.0, cov_matrix=None, samples=None, scatter_in_pct=0.0):
        if b_q025 is None:
            b_q025 = b_mean - 1.96 * b_sd
        if b_q975 is None:
            b_q975 = b_mean + 1.96 * b_sd
        if lambda_q025 is None:
            lambda_q025 = max(0, lambda_mean - 1.96 * lambda_sd)
        if lambda_q975 is None:
            lambda_q975 = lambda_mean + 1.96 * lambda_sd
        if cov_matrix is None:
            cov_matrix = np.diag([lambda_sd**2, b_sd**2])
        return GRResult(
            b_mean=b_mean, b_sd=b_sd, b_q025=b_q025, b_q975=b_q975,
            lambda_mean=lambda_mean, lambda_sd=lambda_sd,
            lambda_q025=lambda_q025, lambda_q975=lambda_q975,
            rho_lb=rho_lb, cov_matrix=cov_matrix, method=method,
            samples=samples, scatter_in_pct=scatter_in_pct, n_events=n_events,
        )


@dataclass
class HierarchicalResult:
    """Results from a multi-zone hierarchical analysis."""
    zone_results: list  # List[GRResult]
    zone_names: list    # List[str]
    mu_b: float
    sigma_b: float
    shrinkage: np.ndarray
    samples: dict


@dataclass
class ZoneData:
    """Binned earthquake data for a single source zone (used by hierarchical models)."""
    name: str
    mi_lo: np.ndarray
    ni: np.ndarray
    ti: np.ndarray
    mn: float
    mx: float
    dm: float

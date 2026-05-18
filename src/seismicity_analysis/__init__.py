"""
SeismicityAnalysis — Python implementation of Bayesian seismicity analysis methods.

Converted from the Julia SeismicityAnalysis.jl package developed for ONR T983.
Implements classical MLE methods (Weichert, PMLM, Tinti, Rhoades), synthetic
catalogue generation, logic tree discretization, moment constraints, and
Bayesian L5 Full Bayesian models via CmdStanPy.

Dependencies:
    numpy, scipy, pandas  — core computation
    cmdstanpy             — Stan interface (for Bayesian models)
"""

from .types import (
    Catalogue,
    CompletenessModel,
    CompletenessBin,
    GRConfig,
    GRResult,
    HierarchicalResult,
    ZoneData,
)
from .conversions import (
    ml2mw,
    mw2ml,
    grunthal_slope,
    grunthal_sigma_mw,
    sigma_conv_ml,
    sigma_total_ml,
)
from .completeness import (
    UK_COMPLETENESS,
    observation_period,
    completeness_bins,
    ml_threshold,
)
from .catalogue import (
    load_catalogue,
    filter_complete,
    bin_magnitudes,
    prepare_for_L5,
)
from .classical import (
    fit_weichert,
    fit_pmlm,
    fit_tinti,
    fit_rhoades,
    estimate_nstar,
    estimate_mstar,
)
from .synthetic import generate_catalogue
from .moment_constraints import (
    seismic_moment,
    moment_release_rate,
    apply_moment_constraint,
    moment_release_sensitivity,
    evaluate_logic_tree,
)
from .discretization import (
    DiscretizationScheme,
    MILLER_RICE,
    HEAVY_TAIL,
    ESM,
    EPT,
    EQUAL_WEIGHT,
    discretize_marginal,
    discretize_joint,
    discretize_five_point,
    build_logic_tree,
)
from .full_bayesian import fit_L5
from .hierarchical import fit_hierarchical, prepare_zone_data
from .negative_binomial import fit_negbin, compare_poisson_negbin, empirical_dispersion

__all__ = [
    # Types
    "Catalogue", "CompletenessModel", "CompletenessBin", "GRConfig", "GRResult",
    "HierarchicalResult", "ZoneData",
    # Conversions
    "ml2mw", "mw2ml", "grunthal_slope", "grunthal_sigma_mw",
    "sigma_conv_ml", "sigma_total_ml",
    # Completeness
    "UK_COMPLETENESS", "observation_period", "completeness_bins", "ml_threshold",
    # Catalogue
    "load_catalogue", "filter_complete", "bin_magnitudes", "prepare_for_L5",
    # Classical
    "fit_weichert", "fit_pmlm", "fit_tinti", "fit_rhoades",
    "estimate_nstar", "estimate_mstar",
    # Synthetic
    "generate_catalogue",
    # Moment constraints
    "seismic_moment", "moment_release_rate", "apply_moment_constraint",
    "moment_release_sensitivity", "evaluate_logic_tree",
    # Discretization
    "DiscretizationScheme", "MILLER_RICE", "HEAVY_TAIL", "ESM", "EPT",
    "EQUAL_WEIGHT", "discretize_marginal", "discretize_joint",
    "discretize_five_point", "build_logic_tree",
    # Bayesian
    "fit_L5", "fit_hierarchical", "prepare_zone_data",
    "fit_negbin", "compare_poisson_negbin", "empirical_dispersion",
]

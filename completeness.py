"""
Completeness models for earthquake catalogues.

Provides the UK default model (Mosca et al. 2020) and functions for computing
observation periods and completeness bins.
"""

import numpy as np
from .types import CompletenessModel, CompletenessBin
from .conversions import mw2ml


UK_COMPLETENESS = CompletenessModel(
    mw_thresholds=[6.5, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5],
    start_years=[1650.0, 1650.0, 1650.0, 1700.0, 1750.0, 1850.0, 1975.0, 1985.0],
    end_year=2023.0,
)


def observation_period(mw, model):
    """
    Return the observation time (years) for a given moment magnitude
    under the completeness model. Searches thresholds in descending order.
    """
    for i, thresh in enumerate(model.mw_thresholds):
        if mw >= thresh:
            return model.end_year - model.start_years[i]
    return 0.0


def completeness_bins(model, mw_min, mw_max):
    """
    Compute completeness bins from a model. Each bin covers a magnitude range
    within [mw_min, mw_max] with its own observation time.

    The scatter-IN region below mw_min is not included here; it is handled
    by the Stan model's integration (which uses T_1 for magnitudes below
    the first completeness bin) and by the synthetic generator separately.

    Returns a list of CompletenessBin.
    """
    bins = []

    # Find unique observation periods within [mw_min, mw_max]
    edges = [mw_min]
    for thresh in model.mw_thresholds:
        if mw_min < thresh < mw_max:
            edges.append(thresh)
    edges.append(mw_max)
    edges = sorted(set(edges))

    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        mid = (lo + hi) / 2
        t = observation_period(mid, model)
        if t > 0:
            bins.append(CompletenessBin(lo, hi, t))

    return bins


def ml_threshold(mw_min, dm_ml=0.1):
    """
    Compute the ML detection threshold corresponding to a Mw analysis threshold.
    Returns a dict with 'ml_threshold', 'ml_min_bin', 'mw_min_effective'.

    The threshold is set at half a bin width below the nearest ML bin centre,
    ensuring events that could round into the analysis window are included.
    """
    ml_at_mw_min = float(mw2ml(mw_min))
    ml_min_bin = round(ml_at_mw_min / dm_ml) * dm_ml
    ml_thresh = ml_min_bin - dm_ml / 2
    from .conversions import ml2mw
    mw_min_eff = float(ml2mw(ml_thresh))
    return {
        "ml_threshold": ml_thresh,
        "ml_min_bin": ml_min_bin,
        "mw_min_effective": mw_min_eff,
    }

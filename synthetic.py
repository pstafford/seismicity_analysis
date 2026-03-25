"""
Synthetic earthquake catalogue generation for validation.

Implements the full forward simulation pipeline: true magnitudes → conversion →
measurement noise → rounding → selection.
"""

import numpy as np
from .types import Catalogue, CompletenessModel, CompletenessBin, LOG10
from .conversions import mw2ml, ml2mw, sigma_conv_ml
from .completeness import completeness_bins, ml_threshold


def generate_catalogue(completeness=None, *, b=1.0, lambda_mw_min=2.0, t_obs=50.0,
                       mw_min=3.0, mw_max=6.5, mw_floor=1.0,
                       sigma_ml=0.25, dm_ml=0.1, seed=None):
    """
    Generate a synthetic earthquake catalogue.

    If `completeness` is None, uses constant completeness with `t_obs`.
    If a CompletenessModel is provided, generates with variable completeness.

    The full forward simulation pipeline:
    1. Draw N ~ Poisson(λ × T) events
    2. Convert true Mw → true ML via inverse Grünthal
    3. Add conversion uncertainty (magnitude-dependent)
    4. Add measurement noise ~ N(0, σ_ml)
    5. Round to nearest dm_ml
    6. Select events with rounded ML ≥ ml_threshold(mw_min)

    Returns a dict with the catalogue and ground truth.
    """
    rng = np.random.default_rng(seed)

    if isinstance(completeness, CompletenessModel):
        return _generate_variable(completeness, rng, b=b, lambda_mw_min=lambda_mw_min,
                                  mw_min=mw_min, mw_max=mw_max, mw_floor=mw_floor,
                                  sigma_ml=sigma_ml, dm_ml=dm_ml)
    else:
        return _generate_constant(rng, b=b, lambda_mw_min=lambda_mw_min, t_obs=t_obs,
                                  mw_min=mw_min, mw_max=mw_max, mw_floor=mw_floor,
                                  sigma_ml=sigma_ml, dm_ml=dm_ml)


def _generate_constant(rng, *, b, lambda_mw_min, t_obs, mw_min, mw_max, mw_floor,
                        sigma_ml, dm_ml):
    beta = b * LOG10
    thresh = ml_threshold(mw_min, dm_ml=dm_ml)
    ml_thresh = thresh["ml_threshold"]

    # Rate at mw_floor
    frac_above_min = ((np.exp(-beta * (mw_min - mw_floor)) - np.exp(-beta * (mw_max - mw_floor))) /
                      (1 - np.exp(-beta * (mw_max - mw_floor))))
    lambda_floor = lambda_mw_min / frac_above_min

    N_total = rng.poisson(lambda_floor * t_obs)

    if N_total == 0:
        empty = np.array([], dtype=float)
        return {
            "catalogue": Catalogue(empty, empty, empty, empty, empty, empty),
            "mw_true": empty, "ml_true": empty,
            "n_total_generated": 0, "n_selected": 0,
            "n_scatter_in": 0, "scatter_in_pct": 0.0,
            "b_true": b, "lambda_true": lambda_mw_min, "t_obs": t_obs,
        }

    # Generate true Mw from truncated exponential
    u = rng.uniform(size=N_total)
    mw_true = mw_floor - np.log(1.0 - u * (1.0 - np.exp(-beta * (mw_max - mw_floor)))) / beta

    # Forward pipeline
    ml_true = np.asarray(mw2ml(mw_true), dtype=float)
    s_conv = np.asarray(sigma_conv_ml(ml_true), dtype=float)
    ml_with_conv = ml_true + rng.normal(0, 1, N_total) * s_conv
    ml_with_noise = ml_with_conv + rng.normal(0, sigma_ml, N_total)
    ml_rounded = np.round(ml_with_noise / dm_ml) * dm_ml

    # Select events above threshold
    selected = ml_rounded >= ml_thresh
    ml_sel = ml_rounded[selected]
    mw_true_sel = mw_true[selected]
    ml_true_sel = ml_true[selected]

    n_scatter_in = int(np.sum(mw_true_sel < mw_min))
    n_selected = len(ml_sel)
    scatter_in_pct = 100.0 * n_scatter_in / n_selected if n_selected > 0 else 0.0

    mw_converted = np.asarray(ml2mw(ml_sel), dtype=float)
    sigma_ml_vec = np.full(n_selected, sigma_ml)
    years = np.sort(rng.uniform(0, t_obs, n_selected))
    lat = np.full(n_selected, np.nan)
    lon = np.full(n_selected, np.nan)

    return {
        "catalogue": Catalogue(ml_sel, mw_converted, sigma_ml_vec, years, lat, lon),
        "mw_true": mw_true_sel, "ml_true": ml_true_sel,
        "n_total_generated": N_total, "n_selected": n_selected,
        "n_scatter_in": n_scatter_in, "scatter_in_pct": scatter_in_pct,
        "b_true": b, "lambda_true": lambda_mw_min, "t_obs": t_obs,
    }


def _generate_variable(completeness, rng, *, b, lambda_mw_min, mw_min, mw_max,
                        mw_floor, sigma_ml, dm_ml):
    beta = b * LOG10
    thresh = ml_threshold(mw_min, dm_ml=dm_ml)
    ml_thresh = thresh["ml_threshold"]

    bins = completeness_bins(completeness, mw_min, mw_max)

    # Add scatter-IN region [mw_floor, mw_min) using T_1 (lowest bin's observation time)
    scatter_in_bin = None
    if mw_floor < mw_min and len(bins) > 0:
        scatter_in_bin = CompletenessBin(mw_floor, mw_min, bins[0].t_obs)

    all_ml = []
    all_mw_true = []
    all_ml_true = []
    all_years = []
    n_scatter_in_total = 0

    gen_bins = ([scatter_in_bin] + bins) if scatter_in_bin is not None else bins
    for bn in gen_bins:
        frac = ((np.exp(-beta * (bn.mw_lo - mw_min)) - np.exp(-beta * (bn.mw_hi - mw_min))) /
                (1 - np.exp(-beta * (mw_max - mw_min))))
        lambda_bin = lambda_mw_min * frac
        N_bin = rng.poisson(max(lambda_bin * bn.t_obs, 0.0))

        if N_bin == 0:
            continue

        # True Mw in [bin.mw_lo, bin.mw_hi]
        u = rng.uniform(size=N_bin)
        mw_true = bn.mw_lo - np.log(1.0 - u * (1.0 - np.exp(-beta * (bn.mw_hi - bn.mw_lo)))) / beta

        # Forward pipeline
        ml_true = np.asarray(mw2ml(mw_true), dtype=float)
        s_conv = np.asarray(sigma_conv_ml(ml_true), dtype=float)
        ml_noisy = ml_true + rng.normal(0, 1, N_bin) * s_conv + rng.normal(0, sigma_ml, N_bin)
        ml_rounded = np.round(ml_noisy / dm_ml) * dm_ml

        sel = ml_rounded >= ml_thresh
        all_ml.extend(ml_rounded[sel])
        all_mw_true.extend(mw_true[sel])
        all_ml_true.extend(ml_true[sel])

        start_year = completeness.end_year - bn.t_obs
        years_bin = rng.uniform(start_year, completeness.end_year, int(sel.sum()))
        all_years.extend(years_bin)

        n_scatter_in_total += int(np.sum(mw_true[sel] < mw_min))

    n_selected = len(all_ml)
    scatter_in_pct = 100.0 * n_scatter_in_total / n_selected if n_selected > 0 else 0.0

    # Sort by year
    all_ml = np.array(all_ml)
    all_mw_true = np.array(all_mw_true)
    all_ml_true = np.array(all_ml_true)
    all_years = np.array(all_years)
    perm = np.argsort(all_years)

    mw_converted = np.asarray(ml2mw(all_ml), dtype=float)
    sigma_ml_vec = np.full(n_selected, sigma_ml)

    return {
        "catalogue": Catalogue(
            all_ml[perm], mw_converted[perm], sigma_ml_vec[perm],
            all_years[perm], np.full(n_selected, np.nan), np.full(n_selected, np.nan),
        ),
        "mw_true": all_mw_true[perm], "ml_true": all_ml_true[perm],
        "n_selected": n_selected, "n_scatter_in": n_scatter_in_total,
        "scatter_in_pct": scatter_in_pct,
        "b_true": b, "lambda_true": lambda_mw_min,
    }

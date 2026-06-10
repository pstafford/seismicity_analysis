"""
Earthquake catalogue loading, filtering, and binning.
"""

import numpy as np
import pandas as pd
from .types import Catalogue, GRConfig, CompletenessModel
from .conversions import ml2mw, mw2ml
from .completeness import observation_period, completeness_bins, ml_threshold


def load_catalogue(filepath, fmt="bgs", min_ml=1.0):
    """
    Load an earthquake catalogue from a CSV file. The 'bgs' format expects columns
    for Year, Month, Day, Latitude, Longitude, and ML (case-insensitive matching).

    Returns a Catalogue.
    """
    df = pd.read_csv(filepath, na_values=["", "NA", "NaN"])

    if fmt == "bgs":
        return _load_bgs(df, min_ml)
    else:
        raise ValueError(f"Unknown format: {fmt}. Supported: 'bgs'")


def _load_bgs(df, min_ml):
    # Case-insensitive column mapping
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl == "year":
            col_map[col] = "year"
        elif cl == "month":
            col_map[col] = "month"
        elif cl == "day":
            col_map[col] = "day"
        elif cl in ("latitude", "lat"):
            col_map[col] = "lat"
        elif cl in ("longitude", "lon", "long"):
            col_map[col] = "lon"
        elif cl == "ml":
            col_map[col] = "ml"
        elif cl == "depth":
            col_map[col] = "depth"
        elif cl in ("errml", "ml_error", "sigma_ml", "erml"):
            col_map[col] = "sigma_ml"

    # Find required columns
    inv_map = {v: k for k, v in col_map.items()}
    if "ml" not in inv_map:
        raise ValueError("No ML column found in catalogue")
    if "year" not in inv_map:
        raise ValueError("No Year column found")

    ml_col = inv_map["ml"]
    year_col = inv_map["year"]
    lat_col = inv_map.get("lat")
    lon_col = inv_map.get("lon")
    sigma_col = inv_map.get("sigma_ml")

    # Filter missing ML and apply minimum
    ml_raw = pd.to_numeric(df[ml_col], errors="coerce")
    valid = ml_raw.notna() & (ml_raw >= min_ml)
    df_filt = df.loc[valid].copy()
    ml_raw = ml_raw[valid]

    ml_arr = ml_raw.values.astype(float)
    mw_arr = np.asarray(ml2mw(ml_arr), dtype=float)

    # Year (fractional if month/day available)
    years = pd.to_numeric(df_filt[year_col], errors="coerce").values.astype(float)
    month_col_name = inv_map.get("month")
    day_col_name = inv_map.get("day")
    if month_col_name is not None and day_col_name is not None:
        months = pd.to_numeric(df_filt[month_col_name], errors="coerce").fillna(6).values.astype(float)
        days = pd.to_numeric(df_filt[day_col_name], errors="coerce").fillna(15).values.astype(float)
        years = years + (months - 1) / 12 + (days - 1) / 365.25

    lat = (pd.to_numeric(df_filt[lat_col], errors="coerce").values.astype(float)
           if lat_col is not None else np.full(len(ml_arr), np.nan))
    lon = (pd.to_numeric(df_filt[lon_col], errors="coerce").values.astype(float)
           if lon_col is not None else np.full(len(ml_arr), np.nan))

    # ML uncertainties
    if sigma_col is not None:
        sigma_ml = pd.to_numeric(df_filt[sigma_col], errors="coerce").values.astype(float)
        for i in range(len(sigma_ml)):
            if np.isnan(sigma_ml[i]) or sigma_ml[i] <= 0:
                sigma_ml[i] = _default_sigma_ml(years[i])
    else:
        sigma_ml = np.array([_default_sigma_ml(y) for y in years])

    return Catalogue(ml_arr, mw_arr, sigma_ml, years, lat, lon)


def _default_sigma_ml(year):
    if year < 1900:
        return 0.5
    if year < 1970:
        return 0.4
    if year < 1990:
        return 0.25
    return 0.15


def filter_complete(cat, model, mw_min=3.0):
    """
    Filter a catalogue to events within the completeness model. Returns events
    with Mw >= mw_min that fall within the complete recording period for their magnitude.
    """
    keep = np.zeros(len(cat.ml), dtype=bool)
    for i in range(len(cat.mw)):
        if cat.mw[i] >= mw_min:
            start_year = _completeness_start(cat.mw[i], model)
            if not np.isnan(start_year) and cat.year[i] >= start_year:
                keep[i] = True
    return Catalogue(
        cat.ml[keep], cat.mw[keep], cat.sigma_ml[keep],
        cat.year[keep], cat.lat[keep], cat.lon[keep],
    )


def _completeness_start(mw, model):
    for i, thresh in enumerate(model.mw_thresholds):
        if mw >= thresh:
            return model.start_years[i]
    return np.nan


def _in_completeness_window(mw, year, model, mw_min):
    """
    Boolean mask selecting events whose (magnitude, year) fall within the complete
    recording window [start_year(m), end_year] of the completeness model.

    Events below ``mw_min`` (the scatter-IN buffer that the L5 model retains) are
    evaluated at ``mw_min``, i.e. they are assumed complete over the same window as
    the lowest analysis bin (T_1). This is consistent with how the L5 likelihood and
    the synthetic generator treat the sub-mw_min region.
    """
    mw = np.asarray(mw, dtype=float)
    year = np.asarray(year, dtype=float)
    mask = np.zeros(len(mw), dtype=bool)
    for i in range(len(mw)):
        m_eval = max(mw[i], mw_min)
        start = _completeness_start(m_eval, model)
        if not np.isnan(start) and start <= year[i] <= model.end_year:
            mask[i] = True
    return mask


def bin_magnitudes(cat, mw_min, mw_max, dm=0.1, model=None, apply_completeness=True):
    """
    Bin catalogue magnitudes into magnitude bins. If a CompletenessModel is provided,
    each bin gets its own observation time; otherwise a single observation time is used.

    When a CompletenessModel is provided and ``apply_completeness`` is True (the
    default), only events that fall within the complete recording window for their
    magnitude (year >= start_year, year <= end_year) are counted. This keeps the
    event counts consistent with the per-bin observation times ``t_obs``; without it,
    historical events from incomplete periods inflate the low-magnitude rates. The
    filter is a no-op for synthetic catalogues (whose events lie within their windows
    by construction) and when no model is supplied (uniform completeness).

    Returns a DataFrame with columns 'm_centre', 'n', 't_obs'.
    """
    edges = np.arange(mw_min, mw_max + dm / 2, dm)
    n_bins = len(edges) - 1
    centres = (edges[:-1] + edges[1:]) / 2
    counts = np.zeros(n_bins, dtype=int)

    for idx, m in enumerate(cat.mw):
        if model is not None and apply_completeness:
            start = _completeness_start(m, model)
            if np.isnan(start) or not (start <= cat.year[idx] <= model.end_year):
                continue
        for j in range(n_bins):
            if edges[j] <= m < edges[j + 1]:
                counts[j] += 1
                break

    if model is not None:
        t_obs = np.array([observation_period(c, model) for c in centres])
    else:
        if len(cat.year) == 0:
            t = 50.0
        else:
            t = max(cat.year.max() - cat.year.min(), 1.0)
        t_obs = np.full(n_bins, t)

    return pd.DataFrame({"m_centre": centres, "n": counts, "t_obs": t_obs})


def prepare_for_L5(cat, config, model, apply_completeness=True):
    """
    Prepare data for the L5 Full Bayesian model. Returns a dict with all
    fields needed by the Stan L5 implementations.

    Events are retained if their reported ML is at or above the detection threshold
    ``ml_threshold(mw_min)`` (which sits below mw_min, deliberately keeping the
    scatter-IN buffer the L5 model requires). When ``apply_completeness`` is True
    (the default), events are additionally required to fall within the complete
    recording window for their magnitude — i.e. year in [start_year(m), end_year].
    Buffer events below mw_min are evaluated at mw_min (the T_1 window). Without this,
    events recorded before a magnitude band became complete are passed to Stan while
    the per-bin observation times ``t_obs`` assume completeness, biasing the rate.
    The filter is a no-op for synthetic catalogues, whose events lie within their
    windows by construction.
    """
    thresh = ml_threshold(config.mw_min, dm_ml=config.dm_ml)

    # Filter to events above ML threshold, within the completeness window
    keep = cat.ml >= thresh["ml_threshold"]
    if apply_completeness:
        keep = keep & _in_completeness_window(cat.mw, cat.year, model, config.mw_min)
    ml_kept = cat.ml[keep]
    sigma_ml_kept = cat.sigma_ml[keep]

    # Completeness bins
    bins = completeness_bins(model, config.mw_min, config.mw_max)

    sigma_round = config.dm_ml / np.sqrt(12)

    return {
        "ml_reported": ml_kept,
        "sigma_ml": sigma_ml_kept,
        "sigma_round": sigma_round,
        "mw_floor": config.mw_floor,
        "mw_min": config.mw_min,
        "mw_max": config.mw_max,
        "ml_threshold": thresh["ml_threshold"],
        "n_comp_bins": len(bins),
        "mw_comp_lo": np.array([b.mw_lo for b in bins]),
        "mw_comp_hi": np.array([b.mw_hi for b in bins]),
        "t_obs": np.array([b.t_obs for b in bins]),
        "n_events": int(keep.sum()),
    }

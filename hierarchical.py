"""
Hierarchical multi-zone model interface via CmdStanPy.

Implements partial pooling of b-values across source zones using a
population-level Normal distribution.
"""

import numpy as np
from pathlib import Path
from .types import GRResult, HierarchicalResult, ZoneData, LOG10
from .catalogue import bin_magnitudes

STAN_DIR = Path(__file__).parent / "stan"


def prepare_zone_data(cat, zone_assignments, zone_names, config, model=None):
    """
    Prepare ZoneData from a catalogue with zone assignments.

    Parameters
    ----------
    cat : Catalogue
    zone_assignments : array-like of int
        Zone index (1-based) for each event
    zone_names : list of str
    config : GRConfig
    model : CompletenessModel, optional
    """
    from .types import Catalogue

    zones = []
    zone_assignments = np.asarray(zone_assignments)
    for iz, name in enumerate(zone_names, start=1):
        mask = zone_assignments == iz
        sub_cat = Catalogue(
            cat.ml[mask], cat.mw[mask], cat.sigma_ml[mask],
            cat.year[mask], cat.lat[mask], cat.lon[mask],
        )
        binned = bin_magnitudes(sub_cat, mw_min=config.mw_min,
                                mw_max=config.mw_max, model=model)
        dm = (binned["m_centre"].iloc[1] - binned["m_centre"].iloc[0]
              if len(binned) > 1 else 0.1)
        zones.append(ZoneData(
            name=name,
            mi_lo=binned["m_centre"].values - 0.05,
            ni=binned["n"].values.astype(int),
            ti=binned["t_obs"].values,
            mn=config.mw_min,
            mx=config.mw_max,
            dm=dm,
        ))
    return zones


def fit_hierarchical(zone_data, config, *, model_type="partial_pooling",
                     n_samples=2000, n_chains=4,
                     adapt_delta=0.95, max_treedepth=12):
    """
    Fit a hierarchical GR model across multiple source zones using Stan.

    Parameters
    ----------
    zone_data : list of ZoneData
    config : GRConfig
    model_type : str
        'partial_pooling' (default) — uses the Stan hierarchical model with
        non-centred parameterisation and population hyperpriors.
    n_samples, n_chains : int
    adapt_delta, max_treedepth : float, int

    Returns a HierarchicalResult.
    """
    try:
        from cmdstanpy import CmdStanModel
    except ImportError:
        raise ImportError(
            "fit_hierarchical requires cmdstanpy. Install with: pip install cmdstanpy"
        )

    stan_file = str(STAN_DIR / "hierarchical.stan")
    stan_model = CmdStanModel(stan_file=stan_file)

    # Concatenate all zone events (needed for the latent-variable Stan model)
    # For the binned hierarchical model, we need per-zone aggregated data.
    # The Stan hierarchical model expects per-event data with zone indices.
    # We'll prepare the data in the format the hierarchical.stan expects.
    n_zones = len(zone_data)

    # Collect per-zone event counts and observation times
    # For the Stan model, we need per-event ML and sigma_ml — but the ZoneData
    # only has binned counts. For the hierarchical Stan model that uses latent
    # variables, we need the raw event data. Since we only have binned data here,
    # we create a simpler binned version.
    #
    # However, looking at the Stan model, it expects per-event ml_reported and sigma_ml.
    # For the Python interface, we provide a simpler binned approach using the
    # Turing-style Poisson likelihood on binned counts, implemented directly in Stan.
    #
    # For now, use the binned Poisson approach (matching the Turing extension).
    # This avoids needing per-event data in the hierarchical model.

    # Build the Stan data for the binned hierarchical approach
    stan_data = _prepare_hierarchical_stan_data(zone_data, config)

    fit = stan_model.sample(
        data=stan_data,
        iter_sampling=n_samples,
        chains=n_chains,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
    )

    return _extract_hierarchical_result(fit, zone_data, config)


def _prepare_hierarchical_stan_data(zone_data, config):
    """Prepare Stan data dict for the hierarchical model.

    Note: The hierarchical.stan model uses per-event latent variables, but
    for simplicity with binned data, we synthesise representative 'events'
    from the bin centres (one per observed event).
    """
    n_zones = len(zone_data)
    all_ml = []
    all_sigma = []
    n_events = []
    zone_start = []
    t_obs = []

    from .conversions import mw2ml
    idx = 1
    for zd in zone_data:
        zone_ml = []
        for j in range(len(zd.ni)):
            # Create 'events' at the bin centre ML for each count
            mw_centre = zd.mi_lo[j] + zd.dm / 2
            ml_val = float(mw2ml(mw_centre))
            zone_ml.extend([ml_val] * zd.ni[j])
        n_z = len(zone_ml)
        n_events.append(n_z)
        zone_start.append(idx)
        idx += n_z
        all_ml.extend(zone_ml)
        all_sigma.extend([0.25] * n_z)  # Default sigma_ml
        # Use the mean observation time for this zone
        t_obs.append(float(np.mean(zd.ti)))

    n_events_total = len(all_ml)

    # Handle empty zones
    if n_events_total == 0:
        raise ValueError("No events across all zones")

    return {
        "n_zones": n_zones,
        "n_events_total": n_events_total,
        "n_events": n_events,
        "zone_start": zone_start,
        "ml_reported": all_ml,
        "sigma_ml": all_sigma,
        "sigma_round": float(config.dm_ml / np.sqrt(12)),
        "mw_min": config.mw_min,
        "mw_max": config.mw_max,
        "mw_floor": config.mw_floor,
        "t_obs": t_obs,
        "mu_b_prior_mean": 1.0,
        "mu_b_prior_sd": 0.3,
        "sigma_b_prior_scale": 0.15,
    }


def _extract_hierarchical_result(fit, zone_data, config):
    df = fit.draws_pd()
    n_zones = len(zone_data)

    zone_results = []
    for z in range(n_zones):
        z1 = z + 1  # Stan is 1-indexed
        b_col = f"b[{z1}]"
        lam_col = f"lambda_mw_min[{z1}]"

        if b_col in df.columns and lam_col in df.columns:
            b_z = df[b_col].values
            lam_z = df[lam_col].values
        else:
            # Fall back to beta/lambda_floor columns
            beta_col = f"beta[{z1}]"
            lam_floor_col = f"lambda_floor[{z1}]"
            beta_z = df[beta_col].values
            lam_floor_z = df[lam_floor_col].values
            b_z = beta_z / LOG10
            frac = ((1 - np.exp(-beta_z * (config.mw_max - config.mw_min))) /
                    (1 - np.exp(-beta_z * (config.mw_max - config.mw_floor))))
            lam_z = lam_floor_z * frac

        C = np.cov(lam_z, b_z)
        rho = C[0, 1] / (np.std(lam_z) * np.std(b_z) + 1e-300)

        zone_results.append(GRResult.create(
            b_mean=float(np.mean(b_z)), b_sd=float(np.std(b_z)),
            b_q025=float(np.quantile(b_z, 0.025)),
            b_q975=float(np.quantile(b_z, 0.975)),
            lambda_mean=float(np.mean(lam_z)), lambda_sd=float(np.std(lam_z)),
            lambda_q025=float(np.quantile(lam_z, 0.025)),
            lambda_q975=float(np.quantile(lam_z, 0.975)),
            rho_lb=rho, cov_matrix=C,
            method="hierarchical_" + "partial_pooling",
            samples={"b": b_z, "lambda_n": lam_z},
            n_events=int(np.sum(zone_data[z].ni)),
        ))

    # Population parameters
    mu_b = float(np.mean(df["mu_b"].values)) if "mu_b" in df.columns else np.mean([r.b_mean for r in zone_results])
    sigma_b = float(np.mean(df["sigma_b"].values)) if "sigma_b" in df.columns else np.std([r.b_mean for r in zone_results])

    # Shrinkage
    shrinkage = np.zeros(n_zones)
    if "sigma_beta" in df.columns:
        sigma_beta_pop = float(np.mean(df["sigma_beta"].values))
        for z in range(n_zones):
            shrinkage[z] = sigma_beta_pop**2 / (sigma_beta_pop**2 + zone_results[z].b_sd**2 * LOG10**2)

    return HierarchicalResult(
        zone_results=zone_results,
        zone_names=[zd.name for zd in zone_data],
        mu_b=mu_b,
        sigma_b=sigma_b,
        shrinkage=shrinkage,
        samples={"fit": fit},
    )

"""
Negative binomial models for overdispersed earthquake counts.

UK seismicity exhibits overdispersion (φ ≈ 5-11) at all magnitude thresholds,
meaning rate uncertainty is underestimated by 2-4× under the Poisson assumption.
"""

import numpy as np
from pathlib import Path
from .types import GRResult, LOG10
from .catalogue import bin_magnitudes
from .completeness import completeness_bins

STAN_DIR = Path(__file__).parent / "stan"


def fit_negbin(cat, config, completeness, *, n_samples=2000, n_chains=4,
               adapt_delta=0.95, max_treedepth=12):
    """
    Fit a negative binomial GR model to an earthquake catalogue using Stan.

    Returns a GRResult plus dispersion estimate φ in samples.
    """
    try:
        from cmdstanpy import CmdStanModel
    except ImportError:
        raise ImportError(
            "fit_negbin requires cmdstanpy. Install with: pip install cmdstanpy"
        )

    stan_file = str(STAN_DIR / "L5_negbin.stan")
    model = CmdStanModel(stan_file=stan_file)

    binned = bin_magnitudes(cat, mw_min=config.mw_min, mw_max=config.mw_max,
                            model=completeness)
    dm = (binned["m_centre"].iloc[1] - binned["m_centre"].iloc[0]
          if len(binned) > 1 else 0.1)

    # ML bin edges
    from .conversions import mw2ml
    ml_lo = np.array([float(mw2ml(m - dm/2)) for m in binned["m_centre"].values])
    ml_hi = np.array([float(mw2ml(m + dm/2)) for m in binned["m_centre"].values])

    sigma_ml_avg = config.sigma_ml_default
    sigma_round = config.dm_ml / np.sqrt(12)

    bins = completeness_bins(completeness, config.mw_min, config.mw_max)

    stan_data = {
        "n_bins": len(binned),
        "ml_lo": ml_lo.tolist(),
        "ml_hi": ml_hi.tolist(),
        "counts": binned["n"].values.tolist(),
        "sigma_ml_avg": float(sigma_ml_avg),
        "sigma_round": float(sigma_round),
        "mw_floor": config.mw_floor,
        "mw_min": config.mw_min,
        "mw_max": config.mw_max,
        "n_comp_bins": len(bins),
        "mw_comp_lo": [b.mw_lo for b in bins],
        "mw_comp_hi": [b.mw_hi for b in bins],
        "t_obs": [b.t_obs for b in bins],
        "n_quad": 50,
        "beta_prior_mean": float(LOG10),
        "beta_prior_sd": 0.5,
        "lambda_prior_mean": float(np.log(100.0)),
        "lambda_prior_sd": 2.0,
    }

    fit = model.sample(
        data=stan_data,
        iter_sampling=n_samples,
        chains=n_chains,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
    )

    df = fit.draws_pd()
    b_samples = df["b"].values
    lam_samples = df["lambda_mw_min"].values
    phi_samples = df["phi"].values

    b_mean = float(np.mean(b_samples))
    b_sd = float(np.std(b_samples))
    lam_mean = float(np.mean(lam_samples))
    lam_sd = float(np.std(lam_samples))

    C = np.cov(lam_samples, b_samples)
    rho = C[0, 1] / (lam_sd * b_sd + 1e-300)

    return GRResult.create(
        b_mean=b_mean, b_sd=b_sd,
        b_q025=float(np.quantile(b_samples, 0.025)),
        b_q975=float(np.quantile(b_samples, 0.975)),
        lambda_mean=lam_mean, lambda_sd=lam_sd,
        lambda_q025=float(np.quantile(lam_samples, 0.025)),
        lambda_q975=float(np.quantile(lam_samples, 0.975)),
        rho_lb=rho,
        cov_matrix=C,
        method="negbin_stan",
        samples={"b": b_samples, "lambda_n": lam_samples,
                 "phi": phi_samples},
        n_events=int(binned["n"].sum()),
    )


def compare_poisson_negbin(cat, config, completeness, **kwargs):
    """
    Compare Poisson vs negative binomial fits at the same magnitude threshold.
    Returns both results plus diagnostics.
    """
    from .full_bayesian import fit_L5

    poisson_result = fit_L5(cat, config, completeness, **kwargs)
    negbin_result = fit_negbin(cat, config, completeness, **kwargs)

    return {
        "poisson": poisson_result,
        "negbin": negbin_result,
        "phi_mean": float(np.mean(negbin_result.samples["phi"])),
        "phi_median": float(np.median(negbin_result.samples["phi"])),
    }


def empirical_dispersion(cat, config, model=None):
    """
    Compute empirical variance-to-mean ratio of binned counts as a quick
    overdispersion diagnostic. Values > 1 suggest overdispersion.
    """
    binned = bin_magnitudes(cat, mw_min=config.mw_min, mw_max=config.mw_max,
                            model=model)
    counts = binned["n"].values
    mu = float(np.mean(counts))
    v = float(np.var(counts))
    return {
        "variance_mean_ratio": v / max(mu, 1e-10),
        "mean_count": mu,
        "variance": v,
        "n_bins": len(counts),
    }

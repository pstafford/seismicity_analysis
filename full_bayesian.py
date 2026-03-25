"""
Full Bayesian L5 model interface via CmdStanPy.

The L5 model uses a thinned Poisson process with latent true magnitudes,
accounting for magnitude conversion, measurement uncertainty, rounding, and selection.
"""

import numpy as np
from pathlib import Path
from .types import GRResult, LOG10

STAN_DIR = Path(__file__).parent / "stan"


def fit_L5(cat, config, completeness, *, n_samples=2000, n_chains=4,
           adapt_delta=0.95, max_treedepth=12):
    """
    Fit the Full Bayesian L5 model to an earthquake catalogue using Stan.

    The L5 model samples latent true ML magnitudes, properly accounting for:
    - ML→Mw conversion uncertainty (Grünthal 2009)
    - Per-event measurement uncertainty
    - Rounding to nearest dm_ml
    - Selection effects (scatter-IN/scatter-OUT)
    - Variable completeness

    Parameters
    ----------
    cat : Catalogue
    config : GRConfig
    completeness : CompletenessModel
    n_samples : int
        Posterior samples per chain (default 2000)
    n_chains : int
        Number of MCMC chains (default 4)
    adapt_delta : float
        Stan adapt_delta (default 0.95)
    max_treedepth : int
        Stan max_treedepth (default 12)

    Returns a GRResult.
    """
    try:
        from cmdstanpy import CmdStanModel
    except ImportError:
        raise ImportError(
            "fit_L5 requires cmdstanpy. Install with: pip install cmdstanpy\n"
            "Then install CmdStan: python -m cmdstanpy.install_cmdstan"
        )

    from .catalogue import prepare_for_L5

    data = prepare_for_L5(cat, config, completeness)

    # Choose constant vs variable model
    has_variable_sigma = not np.allclose(data["sigma_ml"], data["sigma_ml"][0])
    has_variable_completeness = data["n_comp_bins"] > 1

    if has_variable_sigma or has_variable_completeness:
        return _fit_L5_variable(data, config, n_samples, n_chains,
                                adapt_delta, max_treedepth)
    else:
        return _fit_L5_constant(data, config, n_samples, n_chains,
                                adapt_delta, max_treedepth)


def _fit_L5_constant(data, config, n_samples, n_chains, adapt_delta, max_treedepth):
    from cmdstanpy import CmdStanModel

    stan_file = str(STAN_DIR / "L5_constant.stan")
    model = CmdStanModel(stan_file=stan_file)

    stan_data = {
        "N": data["n_events"],
        "ml_reported": data["ml_reported"].tolist(),
        "sigma_ml": float(data["sigma_ml"][0]),
        "sigma_round": float(data["sigma_round"]),
        "mw_min": config.mw_min,
        "mw_max": config.mw_max,
        "mw_floor": data["mw_floor"],
        "ml_threshold": data["ml_threshold"],
        "t_obs": float(data["t_obs"][0]),
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

    return _extract_L5_result(fit, config, data["n_events"], "L5_stan_constant")


def _fit_L5_variable(data, config, n_samples, n_chains, adapt_delta, max_treedepth):
    from cmdstanpy import CmdStanModel

    stan_file = str(STAN_DIR / "L5_variable.stan")
    model = CmdStanModel(stan_file=stan_file)

    stan_data = {
        "N": data["n_events"],
        "ml_reported": data["ml_reported"].tolist(),
        "sigma_ml": data["sigma_ml"].tolist(),
        "sigma_round": float(data["sigma_round"]),
        "mw_min": config.mw_min,
        "mw_max": config.mw_max,
        "mw_floor": data["mw_floor"],
        "ml_threshold": data["ml_threshold"],
        "n_comp_bins": data["n_comp_bins"],
        "mw_comp_lo": data["mw_comp_lo"].tolist(),
        "mw_comp_hi": data["mw_comp_hi"].tolist(),
        "t_obs": data["t_obs"].tolist(),
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

    return _extract_L5_result(fit, config, data["n_events"], "L5_stan_variable")


def _extract_L5_result(fit, config, n_events, method):
    df = fit.draws_pd()
    b_samples = df["b"].values
    lambda_samples = df["lambda_mw_min"].values

    b_mean = float(np.mean(b_samples))
    b_sd = float(np.std(b_samples))
    lam_mean = float(np.mean(lambda_samples))
    lam_sd = float(np.std(lambda_samples))

    C = np.cov(lambda_samples, b_samples)
    rho = C[0, 1] / (lam_sd * b_sd + 1e-300)

    return GRResult.create(
        b_mean=b_mean, b_sd=b_sd,
        b_q025=float(np.quantile(b_samples, 0.025)),
        b_q975=float(np.quantile(b_samples, 0.975)),
        lambda_mean=lam_mean, lambda_sd=lam_sd,
        lambda_q025=float(np.quantile(lambda_samples, 0.025)),
        lambda_q975=float(np.quantile(lambda_samples, 0.975)),
        rho_lb=rho,
        cov_matrix=C,
        method=method,
        samples={"b": b_samples, "lambda_n": lambda_samples},
        n_events=n_events,
    )

"""
Seismic moment release calculations and importance sampling constraints.

Provides analytical integration of moment release over GR distributions
and importance sampling for constraining logic trees.
"""

import numpy as np
from .types import LOG10


def seismic_moment(M):
    """Kanamori seismic moment (N·m) for moment magnitude M: M₀ = 10^(1.5M + 9.05)."""
    return 10.0 ** (1.5 * np.asarray(M) + 9.05)


def moment_release_rate(lambda_n, b, Mmin, Mmax, method="analytical"):
    """
    Annual seismic moment release rate (N·m/year) for a doubly-bounded GR distribution
    with activity rate λn (events/year above Mmin), b-value, and magnitude range [Mmin, Mmax].
    """
    beta = b * LOG10
    if method == "analytical":
        return _moment_rate_analytical(lambda_n, beta, Mmin, Mmax)
    else:
        return _moment_rate_numerical(lambda_n, beta, Mmin, Mmax)


def _moment_rate_analytical(lambda_n, beta, Mmin, Mmax):
    c = 1.5 * np.log(10)  # ≈ 3.4539
    k = 10.0 ** 9.05

    if abs(c - beta) < 1e-10:
        return (lambda_n * k * beta * (Mmax - Mmin) * np.exp(c * Mmin) /
                (1.0 - np.exp(-beta * (Mmax - Mmin))))

    denom = (1.0 - np.exp(-beta * (Mmax - Mmin))) * (c - beta)
    term1 = k * beta * np.exp(beta * Mmin)
    term2 = np.exp((c - beta) * Mmax) - np.exp((c - beta) * Mmin)

    return lambda_n * term1 * term2 / denom


def _moment_rate_numerical(lambda_n, beta, Mmin, Mmax, n_quad=1000):
    dm = (Mmax - Mmin) / n_quad
    m = Mmin + (np.arange(n_quad) + 0.5) * dm
    f_gr = beta * np.exp(-beta * (m - Mmin)) / (1.0 - np.exp(-beta * (Mmax - Mmin)))
    Mdot = np.sum(seismic_moment(m) * f_gr * dm)
    return lambda_n * Mdot


def moment_release_sensitivity(lambda_n, b, Mmin, Mmax):
    """
    Numerical partial derivatives of moment release rate with respect to each parameter.
    Returns normalised elasticities: ε_x = (∂Ṁ/∂x)(x/Ṁ).
    """
    Mdot = moment_release_rate(lambda_n, b, Mmin, Mmax)
    h = 1e-6

    d_lambda = (moment_release_rate(lambda_n + h, b, Mmin, Mmax) - Mdot) / h
    d_b = (moment_release_rate(lambda_n, b + h, Mmin, Mmax) - Mdot) / h
    d_Mmax = (moment_release_rate(lambda_n, b, Mmin, Mmax + h) - Mdot) / h

    return {
        "eps_lambda": d_lambda * lambda_n / Mdot,
        "eps_b": d_b * b / Mdot,
        "eps_Mmax": d_Mmax * Mmax / Mdot,
    }


def apply_moment_constraint(samples, target_Mdot, sigma_log_Mdot, Mmin, Mmax,
                            Mmax_samples=None):
    """
    Apply a moment release constraint to posterior samples via importance sampling.

    Uses a log-normal likelihood on log10(Ṁ) to reweight samples.
    Returns reweighted samples and effective sample size.

    Parameters
    ----------
    samples : dict with keys 'lambda_n' and 'b' (arrays of posterior draws)
    target_Mdot : target moment release rate (N·m/year)
    sigma_log_Mdot : uncertainty in log10(Ṁ)
    """
    n = len(samples["lambda_n"])
    log_target = np.log10(target_Mdot)

    log_Mdot = np.zeros(n)
    for i in range(n):
        mx = Mmax_samples[i] if Mmax_samples is not None else Mmax
        log_Mdot[i] = np.log10(moment_release_rate(samples["lambda_n"][i],
                                                    samples["b"][i], Mmin, mx))

    # Log-normal importance weights
    log_w = -0.5 * ((log_Mdot - log_target) / sigma_log_Mdot) ** 2
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()

    ESS = 1.0 / np.sum(w**2)

    return {
        "weights": w,
        "ESS": ESS,
        "ESS_fraction": ESS / n,
        "log_Mdot": log_Mdot,
        "lambda_n_weighted_mean": np.sum(w * samples["lambda_n"]),
        "b_weighted_mean": np.sum(w * samples["b"]),
        "Mdot_weighted_mean": 10.0 ** np.sum(w * log_Mdot),
    }


def evaluate_logic_tree(branches, target_Mdot, sigma_log_Mdot=0.3, Mmin=3.0):
    """
    Evaluate feasibility of a logic tree against a moment release constraint.

    Parameters
    ----------
    branches : list of dicts with keys 'lambda_n', 'b', 'Mmax', 'weight'
    """
    n = len(branches)
    log_target = np.log10(target_Mdot)

    Mdot = np.array([moment_release_rate(br["lambda_n"], br["b"], Mmin, br["Mmax"])
                     for br in branches])
    log_Mdot = np.log10(Mdot)
    w_orig = np.array([br["weight"] for br in branches])

    log_lik = -0.5 * ((log_Mdot - log_target) / sigma_log_Mdot) ** 2
    lik = np.exp(log_lik - log_lik.max())
    w_adj = w_orig * lik
    w_adj /= w_adj.sum()

    ESS = 1.0 / np.sum(w_adj**2)

    return {
        "Mdot": Mdot,
        "log_Mdot": log_Mdot,
        "weights_original": w_orig,
        "weights_adjusted": w_adj,
        "ESS": ESS,
        "Mdot_mean_original": np.sum(w_orig * Mdot),
        "Mdot_mean_adjusted": np.sum(w_adj * Mdot),
        "log_Mdot_mean_adjusted": np.sum(w_adj * log_Mdot),
    }

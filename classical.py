"""
Classical maximum likelihood estimation methods for Gutenberg-Richter parameters.

Implements Weichert (1980), PMLM/Johnston (1994), Tinti & Mulargia (1985),
Rhoades (1996), and NUREG-2115 N*/M* methods.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from .types import GRResult, LOG10
from .conversions import grunthal_sigma_mw
from .catalogue import bin_magnitudes


# ──────────────────────────────────────────────────────────────────────────────
# Weichert (1980) MLE
# ──────────────────────────────────────────────────────────────────────────────

def fit_weichert(cat, config, model=None):
    """
    Maximum likelihood estimation of GR parameters using the Weichert (1980) method.
    Supports variable completeness via magnitude-dependent observation times.

    Parameters
    ----------
    cat : Catalogue or DataFrame
        If Catalogue, will be binned first. If DataFrame, expects columns
        'm_centre', 'n', 't_obs'.
    config : GRConfig
    model : CompletenessModel, optional

    Returns a GRResult with full covariance matrix.
    """
    from .types import Catalogue
    if isinstance(cat, Catalogue):
        binned = bin_magnitudes(cat, mw_min=config.mw_min, mw_max=config.mw_max,
                                model=model)
    else:
        binned = cat

    mi = binned["m_centre"].values
    ni = binned["n"].values
    ti = binned["t_obs"].values
    mn = config.mw_min
    mx = config.mw_max
    dm = mi[1] - mi[0] if len(mi) > 1 else 0.1
    N = int(ni.sum())

    if N == 0:
        raise ValueError("No events in catalogue for Weichert fitting")

    def negll_beta(beta):
        ebdm = np.exp(-beta * dm)
        denom = 1.0 - np.exp(-beta * (mx - mn))
        pj = (1.0 - ebdm) / denom * np.exp(-beta * (mi - mn))
        # Weichert (1980) profile log-likelihood with λ eliminated:
        #   ℓ(β) = Σ nⱼ log(tⱼ pⱼ) − N log(Σ tⱼ pⱼ)
        tp = ti * pj
        ll = np.sum(ni * np.log(np.maximum(tp, 1e-300))) - N * np.log(np.sum(tp))
        return -ll

    result = minimize_scalar(negll_beta, bounds=(0.5, 5.0), method="bounded")
    beta_hat = result.x
    b_hat = beta_hat / LOG10

    # Compute lambda
    ebdm = np.exp(-beta_hat * dm)
    denom = 1.0 - np.exp(-beta_hat * (mx - mn))
    pj = (1.0 - ebdm) / denom * np.exp(-beta_hat * (mi - mn))
    sum_ti_pi = np.sum(ti * pj)
    lambda_hat = N / sum_ti_pi

    # Fisher information matrix for covariance
    M = _weichert_fisher(beta_hat, lambda_hat, mi, ni, ti, mn, mx, dm)
    C = _safe_inv2x2(M)

    sigma_lambda = np.sqrt(max(C[0, 0], 0.0))
    sigma_beta = np.sqrt(max(C[1, 1], 0.0))
    sigma_b = sigma_beta / LOG10
    rho = C[0, 1] / (sigma_lambda * sigma_beta + 1e-300)

    # Covariance in (λ, b) space
    J = np.array([[1.0, 0.0], [0.0, 1.0 / LOG10]])
    C_lb = J @ C @ J.T

    return GRResult.create(
        b_mean=b_hat, b_sd=sigma_b,
        lambda_mean=lambda_hat, lambda_sd=sigma_lambda,
        rho_lb=rho,
        cov_matrix=C_lb,
        method="weichert",
        n_events=N,
    )


def _weichert_fisher(beta, lam, mi, ni, ti, mn, mx, dm):
    ebdm = np.exp(-beta * dm)
    denom = 1.0 - np.exp(-beta * (mx - mn))

    M11 = np.sum(ni) / lam**2
    M12 = 0.0
    M22 = 0.0

    for j in range(len(mi)):
        pj = (1.0 - ebdm) / denom * np.exp(-beta * (mi[j] - mn))
        dpj = pj * (dm * ebdm / (1.0 - ebdm)
                     - (mx - mn) * np.exp(-beta * (mx - mn)) / denom
                     - (mi[j] - mn))
        M12 += ti[j] * dpj
        if pj > 0:
            M22 += lam * ti[j] * (dpj**2 / pj - _d2pj_dbeta2(beta, mi[j], mn, mx, dm))

    return np.array([[M11, M12], [M12, M22]])


def _d2pj_dbeta2(beta, mj, mn, mx, dm):
    h = 1e-5
    vals = []
    for b in [beta - h, beta, beta + h]:
        ebdm = np.exp(-b * dm)
        denom = 1.0 - np.exp(-b * (mx - mn))
        vals.append((1.0 - ebdm) / denom * np.exp(-b * (mj - mn)))
    return (vals[2] - 2 * vals[1] + vals[0]) / h**2


def _safe_inv2x2(M):
    det_M = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det_M) < 1e-30:
        return np.array([[np.inf, 0.0], [0.0, np.inf]])
    return np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]]) / det_M


# ──────────────────────────────────────────────────────────────────────────────
# PMLM (Johnston et al. 1994)
# ──────────────────────────────────────────────────────────────────────────────

def fit_pmlm(cat, config, prior_b=1.0, prior_sigma=0.087, model=None):
    """
    Penalised maximum likelihood method (Johnston et al. 1994).
    Mathematically equivalent to MAP estimation with Normal prior on β.

    Default prior_sigma=0.087 corresponds to UK practice weight W=25.
    """
    from .types import Catalogue
    if isinstance(cat, Catalogue):
        binned = bin_magnitudes(cat, mw_min=config.mw_min, mw_max=config.mw_max,
                                model=model)
    else:
        binned = cat

    mi = binned["m_centre"].values
    ni = binned["n"].values
    ti = binned["t_obs"].values
    mn = config.mw_min
    mx = config.mw_max
    dm = mi[1] - mi[0] if len(mi) > 1 else 0.1
    N = int(ni.sum())

    if N == 0:
        raise ValueError("No events in catalogue for PMLM fitting")

    beta_prior = prior_b * LOG10
    sigma_beta_prior = prior_sigma * LOG10
    w = 1.0 / sigma_beta_prior**2

    def negpll(beta):
        ebdm = np.exp(-beta * dm)
        denom = 1.0 - np.exp(-beta * (mx - mn))
        pj = (1.0 - ebdm) / denom * np.exp(-beta * (mi - mn))
        tp = ti * pj
        ll = np.sum(ni * np.log(np.maximum(tp, 1e-300))) - N * np.log(np.sum(tp))
        penalty = -0.5 * w * (beta - beta_prior)**2
        return -(ll + penalty)

    result = minimize_scalar(negpll, bounds=(0.5, 5.0), method="bounded")
    beta_hat = result.x
    b_hat = beta_hat / LOG10

    # Lambda
    ebdm = np.exp(-beta_hat * dm)
    denom = 1.0 - np.exp(-beta_hat * (mx - mn))
    pj = (1.0 - ebdm) / denom * np.exp(-beta_hat * (mi - mn))
    sum_ti_pi = np.sum(ti * pj)
    lambda_hat = N / sum_ti_pi

    # Fisher information with penalty
    M = _weichert_fisher(beta_hat, lambda_hat, mi, ni, ti, mn, mx, dm)
    M[1, 1] += w  # Add prior precision
    C = _safe_inv2x2(M)

    sigma_lambda = np.sqrt(max(C[0, 0], 0.0))
    sigma_beta = np.sqrt(max(C[1, 1], 0.0))
    sigma_b = sigma_beta / LOG10
    rho = C[0, 1] / (sigma_lambda * sigma_beta + 1e-300)

    J = np.array([[1.0, 0.0], [0.0, 1.0 / LOG10]])
    C_lb = J @ C @ J.T

    return GRResult.create(
        b_mean=b_hat, b_sd=sigma_b,
        lambda_mean=lambda_hat, lambda_sd=sigma_lambda,
        rho_lb=rho,
        cov_matrix=C_lb,
        method="pmlm",
        n_events=N,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tinti & Mulargia (1985) Rate Correction
# ──────────────────────────────────────────────────────────────────────────────

def fit_tinti(cat, config, sigma_mean=None, model=None):
    """
    Tinti & Mulargia (1985) rate correction for magnitude uncertainty.
    Corrects the activity rate: λ_corrected = λ_MLE × exp(-β² σ² / 2).

    If sigma_mean is not provided, uses the mean of per-event uncertainties in Mw space.
    """
    base = fit_weichert(cat, config, model=model)

    if sigma_mean is not None:
        sigma = sigma_mean
    else:
        sigma = float(np.mean(grunthal_sigma_mw(cat.ml)))

    beta = base.b_mean * LOG10
    correction = np.exp(-beta**2 * sigma**2 / 2)
    lambda_corrected = base.lambda_mean * correction

    return GRResult.create(
        b_mean=base.b_mean, b_sd=base.b_sd,
        lambda_mean=lambda_corrected, lambda_sd=base.lambda_sd * correction,
        rho_lb=base.rho_lb,
        cov_matrix=base.cov_matrix,
        method="tinti",
        n_events=base.n_events,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Rhoades (1996) Iterative b-value Correction
# ──────────────────────────────────────────────────────────────────────────────

def fit_rhoades(cat, config, sigma_events=None, max_iter=100, tol=1e-6, model=None):
    """
    Rhoades (1996) iterative b-value correction for magnitude uncertainty.
    Corrects the mean magnitude: m̄_corrected = m̄_obs - β σ².

    If sigma_events is not provided, uses per-event Grünthal uncertainties.
    """
    keep = (cat.mw >= config.mw_min) & (cat.mw < config.mw_max)
    mw_obs = cat.mw[keep]
    N = len(mw_obs)
    if N == 0:
        raise ValueError("No events in analysis window")

    if sigma_events is not None:
        sigma = sigma_events[keep]
    else:
        sigma = grunthal_sigma_mw(cat.ml[keep])

    mn = config.mw_min
    mx = config.mw_max
    m_bar_obs = float(np.mean(mw_obs))
    sigma2_mean = float(np.mean(sigma**2))

    def gr_mean(beta):
        return mn + 1.0 / beta - (mx - mn) * np.exp(-beta * (mx - mn)) / (1.0 - np.exp(-beta * (mx - mn)))

    # Iterative correction
    beta = LOG10  # initial guess (b=1)
    for _ in range(max_iter):
        m_bar_corrected = m_bar_obs - beta * sigma2_mean
        target = m_bar_corrected

        # Bisection to solve gr_mean(beta_new) = target
        beta_lo, beta_hi = 0.1, 10.0
        beta_new = beta
        for _ in range(100):
            beta_mid = (beta_lo + beta_hi) / 2
            if gr_mean(beta_mid) - target > 0:
                beta_lo = beta_mid
            else:
                beta_hi = beta_mid
            if beta_hi - beta_lo < 1e-10:
                beta_new = beta_mid
                break
            beta_new = beta_mid

        if abs(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new

    b = beta / LOG10
    base = fit_weichert(cat, config, model=model)

    return GRResult.create(
        b_mean=b, b_sd=base.b_sd,
        lambda_mean=base.lambda_mean, lambda_sd=base.lambda_sd,
        rho_lb=base.rho_lb,
        cov_matrix=base.cov_matrix,
        method="rhoades",
        n_events=N,
    )


# ──────────────────────────────────────────────────────────────────────────────
# NUREG-2115 M* and N* Methods
# ──────────────────────────────────────────────────────────────────────────────

def estimate_mstar(m_obs, sigma_obs, beta=LOG10):
    """
    NUREG-2115 M* method: shift each magnitude by -β σ² to correct for
    magnitude uncertainty bias.
    """
    m_obs = np.asarray(m_obs)
    sigma_obs = np.asarray(sigma_obs)
    return m_obs - beta * sigma_obs**2


def estimate_nstar(m_obs, sigma_obs, beta=LOG10):
    """
    NUREG-2115 N* method: weight each event by exp(-β² σ² / 2) to correct for
    magnitude uncertainty bias in event counts.
    """
    sigma_obs = np.asarray(sigma_obs)
    return np.exp(-beta**2 * sigma_obs**2 / 2)

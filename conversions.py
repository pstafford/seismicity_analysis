"""
Magnitude conversion functions based on Grünthal et al. (2009).

Polynomial ML→Mw conversion for Central Europe, with associated uncertainty model.
"""

import numpy as np


def ml2mw(ml):
    """
    Convert local magnitude ML to moment magnitude Mw using Grünthal et al. (2009).

        Mw = 0.0376 ML² + 0.646 ML + 0.53
    """
    ml = np.asarray(ml, dtype=float)
    return 0.0376 * ml**2 + 0.646 * ml + 0.53


def mw2ml(mw):
    """
    Convert moment magnitude Mw to local magnitude ML (inverse of Grünthal 2009).
    Uses the quadratic formula on 0.0376 ML² + 0.646 ML + (0.53 - Mw) = 0.
    """
    mw = np.asarray(mw, dtype=float)
    a = 0.0376
    b = 0.646
    c = 0.53 - mw
    disc = np.maximum(b**2 - 4 * a * c, 0.0)
    return (-b + np.sqrt(disc)) / (2 * a)


def grunthal_slope(ml):
    """Derivative d(Mw)/d(ML) of the Grünthal (2009) conversion."""
    ml = np.asarray(ml, dtype=float)
    return 2 * 0.0376 * ml + 0.646


def grunthal_sigma_mw(ml):
    """
    Conversion uncertainty σ(Mw|ML) from Grünthal (2009) Annex 4 polynomial variance model.
    Clamped to [0.29, 0.34] following the original specification.
    """
    ml = np.asarray(ml, dtype=float)
    var_mw = (0.97 * ml**4 - 12.4 * ml**3 + 58.4 * ml**2 - 120.0 * ml + 921.0) * 1e-4
    sigma = np.sqrt(np.maximum(var_mw, 0.0))
    return np.clip(sigma, 0.29, 0.34)


def sigma_conv_ml(ml):
    """Conversion uncertainty propagated to ML space: σ_conv(ML) = σ(Mw|ML) / |d(Mw)/d(ML)|."""
    ml = np.asarray(ml, dtype=float)
    return grunthal_sigma_mw(ml) / np.abs(grunthal_slope(ml))


def sigma_total_ml(ml, sigma_ml=0.25, dm_ml=0.1):
    """
    Total ML uncertainty combining conversion, measurement, and rounding components:

        σ_total = √(σ_conv² + σ_ml² + σ_round²)

    where σ_round = dm_ml / √12 (uniform rounding uncertainty).
    """
    ml = np.asarray(ml, dtype=float)
    sigma_round = dm_ml / np.sqrt(12)
    s_conv = sigma_conv_ml(ml)
    return np.sqrt(s_conv**2 + sigma_ml**2 + sigma_round**2)

"""
SeismicityAnalysis — Quickstart Example

Minimal working example: generate synthetic data, fit with classical methods,
and compare results.
"""

import sys
sys.path.insert(0, "..")

from seismicity_analysis import (
    generate_catalogue, GRConfig,
    fit_weichert, fit_pmlm, fit_tinti,
)

# ── Generate a synthetic catalogue ────────────────────────────────────────────
syn = generate_catalogue(
    b=1.0,             # True b-value
    lambda_mw_min=5.0, # 5 events/year above Mw 3.0
    t_obs=50.0,        # 50 years of observation
    mw_min=3.0,        # Analysis threshold
    mw_max=6.5,        # Upper magnitude bound
    sigma_ml=0.25,     # Measurement uncertainty
    seed=42,
)

print(f"Generated {syn['n_selected']} events "
      f"({syn['n_scatter_in']} scatter-IN, {syn['scatter_in_pct']:.1f}%)")

# ── Fit with classical methods ────────────────────────────────────────────────
config = GRConfig(mw_min=3.0, mw_max=6.5)

# Weichert MLE
weichert = fit_weichert(syn["catalogue"], config)
print(f"\nWeichert MLE:")
print(f"  b = {weichert.b_mean:.3f} ± {weichert.b_sd:.3f}")
print(f"  λ = {weichert.lambda_mean:.2f} ± {weichert.lambda_sd:.2f}")

# PMLM (Johnston 1994, UK practice)
pmlm = fit_pmlm(syn["catalogue"], config, prior_b=1.0, prior_sigma=0.087)
print(f"\nPMLM (W=25):")
print(f"  b = {pmlm.b_mean:.3f} ± {pmlm.b_sd:.3f}")
print(f"  λ = {pmlm.lambda_mean:.2f} ± {pmlm.lambda_sd:.2f}")

# Tinti correction
tinti = fit_tinti(syn["catalogue"], config, sigma_mean=0.3)
print(f"\nTinti-Mulargia:")
print(f"  b = {tinti.b_mean:.3f} (same as Weichert)")
print(f"  λ = {tinti.lambda_mean:.2f} (corrected for uncertainty)")

# Compare
print(f"\nTrue values: b = {syn['b_true']}, λ = {syn['lambda_true']}")

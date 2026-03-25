"""
SeismicityAnalysis — Bayesian L5 Full Bayesian Example

Demonstrates fitting the L5 model via CmdStanPy.
Requires: pip install cmdstanpy
Then: python -m cmdstanpy.install_cmdstan
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from seismicity_analysis import (
    generate_catalogue, GRConfig, fit_L5, fit_weichert,
    UK_COMPLETENESS,
)

# ── Generate synthetic data ───────────────────────────────────────────────────
syn = generate_catalogue(
    UK_COMPLETENESS,
    b=1.0,
    lambda_mw_min=5.0,
    mw_min=3.0,
    mw_max=6.5,
    sigma_ml=0.25,
    # seed=42,
)
print(f"Generated {syn['n_selected']} events ({syn['scatter_in_pct']:.1f}% scatter-IN)")

config = GRConfig(mw_min=3.0, mw_max=6.5)

# ── Classical comparison ──────────────────────────────────────────────────────
weichert = fit_weichert(syn["catalogue"], config, model=UK_COMPLETENESS)
print(f"\nWeichert:  b = {weichert.b_mean:.3f} ± {weichert.b_sd:.3f}, "
      f"λ = {weichert.lambda_mean:.2f} ± {weichert.lambda_sd:.2f}")

# ── Bayesian L5 (Stan) ───────────────────────────────────────────────────────
print("\nFitting L5 model via Stan (this may take a few minutes)...")
l5 = fit_L5(syn["catalogue"], config, UK_COMPLETENESS,
            n_samples=1000, n_chains=4)

print(f"\nL5 (Stan): b = {l5.b_mean:.3f} ± {l5.b_sd:.3f}")
print(f"           λ = {l5.lambda_mean:.2f} ± {l5.lambda_sd:.2f}")
print(f"           ρ(λ,b) = {l5.rho_lb:.3f}")
print(f"           95% CI(b) = [{l5.b_q025:.3f}, {l5.b_q975:.3f}]")

print(f"\nTrue values: b = {syn['b_true']}, λ = {syn['lambda_true']}")

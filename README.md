# seismicity_analysis

Bayesian and classical methods for calibrating earthquake magnitude-frequency distributions, with applications to seismic hazard analysis at nuclear sites.

Developed for the UK Office for Nuclear Regulation (ONR) under project T983: *Improved Approaches for Modelling UK Seismicity*.

## Installation

```bash
pip install git+https://github.com/pstafford/seismicity_analysis.git
```

For Bayesian models (requires CmdStan):

```bash
pip install "seismicity-analysis[stan] @ git+https://github.com/pstafford/seismicity_analysis.git"
```

## Quick Start

```python
from seismicity_analysis import (
    generate_catalogue, GRConfig,
    fit_weichert, fit_pmlm, fit_tinti,
)

# Generate synthetic catalogue
syn = generate_catalogue(
    b=1.0, lambda_mw_min=5.0, t_obs=50.0,
    mw_min=3.0, mw_max=6.5, sigma_ml=0.25, seed=42,
)

# Fit with classical methods
config = GRConfig(mw_min=3.0, mw_max=6.5)
weichert = fit_weichert(syn["catalogue"], config)
pmlm = fit_pmlm(syn["catalogue"], config, prior_b=1.0, prior_sigma=0.087)

# Build logic tree preserving (lambda, b) correlation
tree = build_logic_tree(pmlm, scheme=HEAVY_TAIL)
```

## Full Bayesian L5 Model

The L5 model is a thinned Poisson process that properly accounts for magnitude conversion, measurement uncertainty, rounding, and selection effects. It requires CmdStanPy:

```python
from seismicity_analysis import GRConfig, fit_L5, UK_COMPLETENESS
from seismicity_analysis.catalogue import load_catalogue

cat = load_catalogue("bgs_catalogue.csv")
config = GRConfig(mw_min=3.0, mw_max=6.5, mw_floor=1.0)
result = fit_L5(cat, config, UK_COMPLETENESS, n_samples=2000, n_chains=4)
```

A companion Julia implementation is available at [SeismicityAnalysis.jl](https://github.com/pstafford/SeismicityAnalysis.jl), which also provides Turing.jl and R/rstan backends.

## Available Methods

### Classical (no MCMC dependencies)

| Method | Function | Reference |
|--------|----------|-----------|
| Weichert MLE | `fit_weichert` | Weichert (1980) |
| Penalised MLE | `fit_pmlm` | Johnston et al. (1994) |
| Rate correction | `fit_tinti` | Tinti & Mulargia (1985) |
| b-value correction | `fit_rhoades` | Rhoades (1996) |
| M* magnitude shift | `estimate_mstar` | NUREG-2115 |
| N* event weighting | `estimate_nstar` | NUREG-2115 |

### Bayesian (require `cmdstanpy`)

| Method | Function | Description |
|--------|----------|-------------|
| Full Bayesian L5 | `fit_L5` | Marginalised thinned Poisson |
| Hierarchical | `fit_hierarchical` | Multi-zone partial pooling |
| Negative Binomial | `fit_negbin` | Overdispersion model |

### Utilities

| Function | Description |
|----------|-------------|
| `ml2mw`, `mw2ml` | Magnitude conversion (Grunthal 2009) |
| `generate_catalogue` | Synthetic data with full forward simulation |
| `build_logic_tree` | Discretize posterior for PSHA logic trees |
| `moment_release_rate` | Analytical GR moment integration |
| `apply_moment_constraint` | Importance sampling for moment constraints |
| `discretize_joint` | Bivariate discretization preserving correlation |

## Discretization Schemes

Five schemes are provided for discretizing continuous distributions into logic tree branches:

| Scheme | z | Tail weight | Best for |
|--------|---|-------------|----------|
| `MILLER_RICE` | 1.73 | 17% | Mean hazard |
| `HEAVY_TAIL` | 1.03 | 94% | Regulatory fractiles (P84) |
| `ESM` | 1.28 | 60% | Balanced |
| `EPT` | 1.65 | 37% | Extreme percentiles |
| `EQUAL_WEIGHT` | 1.22 | 67% | Equal branch weights |

**Key finding:** Miller-Rice (standard practice) underestimates regulatory fractiles by 20-40%.

## Key Scientific Findings

1. **PMLM ≡ MAP:** Johnston's penalised MLE with weight W is mathematically equivalent to MAP estimation with Normal prior σ_b = 1/√W.

2. **Correlation matters:** ρ(λ, b) ≈ +0.4 at the catalogue threshold but ≈ -0.8 at hazard-relevant magnitudes. Ignoring this overestimates uncertainty.

3. **Overdispersion:** UK seismicity is overdispersed (φ ≈ 5-11), meaning Poisson-based rate uncertainties are underestimated by 2-4×.

4. **Eddington bias protection:** The UK practice of ~3σ completeness buffer provides <2% residual bias.

## Project Structure

```
seismicity_analysis/
├── types.py              # Core data types (Catalogue, GRConfig, GRResult)
├── conversions.py        # ML↔Mw conversion (Grünthal 2009)
├── completeness.py       # Completeness models (UK default)
├── catalogue.py          # Load/filter/bin earthquake catalogues
├── classical.py          # Weichert, PMLM, Tinti, Rhoades, N*/M*
├── full_bayesian.py      # L5 interface (CmdStanPy backend)
├── hierarchical.py       # Multi-zone partial pooling
├── negative_binomial.py  # NegBin overdispersion models
├── moment_constraints.py # Moment release & importance sampling
├── discretization.py     # Logic tree discretization
├── synthetic.py          # Synthetic catalogue generation
stan/
├── L5_constant.stan      # Stan model files
├── L5_variable.stan
├── L5_negbin.stan
└── hierarchical.stan
tests/
├── test_classical.py
├── test_conversions.py
├── test_discretization.py
└── test_synthetic.py
examples/
├── quickstart.py
└── bayesian_L5.py
```

## References

- Stafford, P.J. (2026). *Improved Approaches for Modelling UK Seismicity*. ONR Report T983.
- Weichert, D.H. (1980). Estimation of the earthquake recurrence parameters for unequal observation periods for different magnitudes. *BSSA*, 70(4), 1337-1346.
- Johnston, A.C. et al. (1994). *The Earthquakes of Stable Continental Regions*. EPRI TR-102261.
- Grünthal, G. et al. (2009). *European Macroseismic Scale 1998 (EMS-98)*. Cahiers du Centre Européen de Géodynamique et de Séismologie, Vol. 15.

## License

This software was developed for the UK Office for Nuclear Regulation. See the project report for terms of use.

## Citation

```bibtex
@techreport{Stafford2026,
  author = {Stafford, Peter J},
  title = {Improved Approaches for Modelling {UK} Seismicity},
  institution = {Office for Nuclear Regulation},
  year = {2026},
  number = {T983}
}
```

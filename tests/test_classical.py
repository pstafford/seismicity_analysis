"""Tests for classical MLE methods."""

import numpy as np
import pytest
from seismicity_analysis.types import Catalogue, GRConfig, LOG10
from seismicity_analysis.classical import (
    fit_weichert, fit_pmlm, fit_tinti, fit_rhoades,
    estimate_nstar, estimate_mstar,
)
from seismicity_analysis.completeness import UK_COMPLETENESS


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    b_true = 1.0
    beta_true = b_true * LOG10
    lambda_true = 5.0
    mw_min = 3.0
    mw_max = 6.5
    t_obs = 50.0

    N = rng.poisson(lambda_true * t_obs)
    u = rng.uniform(size=N)
    mw = mw_min - np.log(1.0 - u * (1 - np.exp(-beta_true * (mw_max - mw_min)))) / beta_true
    years = np.sort(rng.uniform(1973.0, 2023.0, N))

    cat = Catalogue(mw, mw, np.full(N, 0.25), years,
                    np.full(N, np.nan), np.full(N, np.nan))
    config = GRConfig(mw_min=mw_min, mw_max=mw_max)
    return cat, config, N


class TestWeichert:
    def test_basic(self, synthetic_data):
        cat, config, N = synthetic_data
        result = fit_weichert(cat, config)
        assert result.method == "weichert"
        assert result.n_events == N
        assert 0.5 < result.b_mean < 1.5
        assert 2.0 < result.lambda_mean < 10.0
        assert result.b_sd > 0
        assert result.lambda_sd > 0
        assert result.cov_matrix.shape == (2, 2)


class TestPMLM:
    def test_smaller_uncertainty(self, synthetic_data):
        cat, config, _ = synthetic_data
        pmlm = fit_pmlm(cat, config, prior_b=1.0, prior_sigma=0.087)
        weichert = fit_weichert(cat, config)
        assert pmlm.method == "pmlm"
        assert pmlm.b_sd <= weichert.b_sd

    def test_weak_prior_equals_weichert(self, synthetic_data):
        cat, config, _ = synthetic_data
        weak = fit_pmlm(cat, config, prior_b=1.0, prior_sigma=10.0)
        weichert = fit_weichert(cat, config)
        assert abs(weak.b_mean - weichert.b_mean) < 0.01


class TestTinti:
    def test_rate_reduction(self, synthetic_data):
        cat, config, _ = synthetic_data
        tinti = fit_tinti(cat, config, sigma_mean=0.3)
        weichert = fit_weichert(cat, config)
        assert tinti.method == "tinti"
        assert tinti.lambda_mean < weichert.lambda_mean


class TestRhoades:
    def test_basic(self, synthetic_data):
        cat, config, _ = synthetic_data
        sigma_small = np.full(len(cat.mw), 0.1)
        result = fit_rhoades(cat, config, sigma_events=sigma_small)
        assert result.method == "rhoades"
        assert 0.5 < result.b_mean < 2.0


class TestNstarMstar:
    def test_nstar(self, synthetic_data):
        cat, _, N = synthetic_data
        sigma_obs = np.full(N, 0.3)
        nstar = estimate_nstar(cat.mw, sigma_obs)
        assert np.all(nstar > 0)
        assert np.all(nstar <= 1)

    def test_mstar(self, synthetic_data):
        cat, _, N = synthetic_data
        sigma_obs = np.full(N, 0.3)
        mstar = estimate_mstar(cat.mw, sigma_obs)
        assert np.all(mstar < cat.mw)


class TestVariableCompleteness:
    def test_with_completeness(self, synthetic_data):
        cat, config, _ = synthetic_data
        result = fit_weichert(cat, config, model=UK_COMPLETENESS)
        assert result.method == "weichert"
        assert result.n_events > 0

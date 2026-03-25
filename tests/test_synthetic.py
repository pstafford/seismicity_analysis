"""Tests for synthetic catalogue generation."""

import numpy as np
import pytest
from seismicity_analysis.synthetic import generate_catalogue
from seismicity_analysis.completeness import UK_COMPLETENESS, ml_threshold
from seismicity_analysis.classical import fit_weichert
from seismicity_analysis.types import GRConfig


class TestConstantCompleteness:
    def test_basic(self):
        result = generate_catalogue(b=1.0, lambda_mw_min=5.0, t_obs=50.0,
                                    mw_min=3.0, mw_max=6.5, seed=123)
        assert result["n_selected"] > 0
        assert result["b_true"] == 1.0
        assert result["lambda_true"] == 5.0
        assert len(result["catalogue"].ml) == result["n_selected"]
        assert len(result["mw_true"]) == result["n_selected"]

        thresh = ml_threshold(3.0)
        assert np.all(result["catalogue"].ml >= thresh["ml_threshold"])


class TestScatterIN:
    def test_tracking(self):
        result = generate_catalogue(b=1.0, lambda_mw_min=5.0, t_obs=100.0,
                                    mw_min=3.0, mw_max=6.5, mw_floor=1.0, seed=456)
        assert result["scatter_in_pct"] >= 0
        assert result["n_scatter_in"] >= 0


class TestVariableCompleteness:
    def test_basic(self):
        result = generate_catalogue(UK_COMPLETENESS,
                                    b=1.0, lambda_mw_min=5.0,
                                    mw_min=3.0, mw_max=6.5, seed=789)
        assert result["n_selected"] > 0
        assert len(result["catalogue"].ml) == result["n_selected"]


class TestReproducibility:
    def test_same_seed(self):
        r1 = generate_catalogue(b=1.0, lambda_mw_min=5.0, t_obs=50.0, seed=42)
        r2 = generate_catalogue(b=1.0, lambda_mw_min=5.0, t_obs=50.0, seed=42)
        assert r1["n_selected"] == r2["n_selected"]
        np.testing.assert_allclose(r1["catalogue"].ml, r2["catalogue"].ml)


class TestParameterRecovery:
    def test_large_clean_sample(self):
        result = generate_catalogue(b=1.0, lambda_mw_min=5.0, t_obs=200.0,
                                    mw_min=3.0, mw_max=6.5, mw_floor=3.0,
                                    sigma_ml=0.0, dm_ml=0.001, seed=999)
        if result["n_selected"] > 50:
            config = GRConfig(mw_min=3.0, mw_max=6.5)
            fit = fit_weichert(result["catalogue"], config)
            assert 0.5 < fit.b_mean < 1.5

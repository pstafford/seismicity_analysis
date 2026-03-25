"""Tests for magnitude conversion functions."""

import numpy as np
import pytest
from seismicity_analysis.conversions import (
    ml2mw, mw2ml, grunthal_slope, grunthal_sigma_mw, sigma_total_ml,
)


class TestMl2mw:
    def test_known_values(self):
        assert ml2mw(0.0) == pytest.approx(0.53)
        assert ml2mw(3.0) == pytest.approx(0.0376 * 9 + 0.646 * 3 + 0.53)
        assert ml2mw(5.0) == pytest.approx(0.0376 * 25 + 0.646 * 5 + 0.53)

    def test_vector(self):
        ml = [2.0, 3.0, 4.0]
        mw = ml2mw(ml)
        assert len(mw) == 3
        assert np.all(np.abs(mw - ml) < 1.0)


class TestRoundTrip:
    def test_roundtrip(self):
        for ml in np.arange(1.0, 7.5, 0.5):
            assert mw2ml(ml2mw(ml)) == pytest.approx(ml, abs=1e-10)


class TestGrunthalSlope:
    def test_known_values(self):
        assert grunthal_slope(0.0) == pytest.approx(0.646)
        assert grunthal_slope(3.0) == pytest.approx(2 * 0.0376 * 3 + 0.646)

    def test_increasing(self):
        assert grunthal_slope(5.0) > grunthal_slope(3.0)


class TestConversionUncertainty:
    def test_clamped(self):
        for ml in np.arange(0.0, 8.5, 0.5):
            sigma = grunthal_sigma_mw(ml)
            assert 0.29 <= sigma <= 0.34


class TestTotalUncertainty:
    def test_larger_than_measurement(self):
        sigma = sigma_total_ml(3.0, sigma_ml=0.25, dm_ml=0.1)
        assert sigma > 0.25
        assert sigma < 1.0

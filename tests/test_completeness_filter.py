"""Tests for the completeness time-window (year) filtering in catalogue preparation.

Regression tests for the bug where prepare_for_L5 / bin_magnitudes filtered events
by magnitude only and ignored the completeness start year, inflating counts/rates on
real catalogues with historical (incomplete-period) events.
"""

import numpy as np
import pytest

from seismicity_analysis.types import Catalogue, GRConfig
from seismicity_analysis.completeness import UK_COMPLETENESS, ml_threshold
from seismicity_analysis.catalogue import prepare_for_L5, bin_magnitudes
from seismicity_analysis.synthetic import generate_catalogue


def _make_catalogue(ml, year):
    """Build a Catalogue from ML and year arrays (mw via Grünthal)."""
    from seismicity_analysis.conversions import ml2mw
    ml = np.asarray(ml, dtype=float)
    year = np.asarray(year, dtype=float)
    mw = np.asarray(ml2mw(ml), dtype=float)
    sigma = np.full(len(ml), 0.2)
    nan = np.full(len(ml), np.nan)
    return Catalogue(ml, mw, sigma, year, nan, nan)


class TestPrepareForL5Window:
    def test_drops_pre_completeness_events(self):
        # mw_min=3.0 -> ML threshold ~3.15; band Mw 3.0-3.5 complete from 1975.
        # Two events at ML 4.0 (Mw ~3.7, band complete from 1850): one in window,
        # one before. Two buffer events at ML 3.2 (Mw ~2.95) evaluated at mw_min
        # (complete from 1975): one in window, one before.
        ml = [4.0, 4.0, 3.2, 3.2]
        year = [1900.0, 1800.0, 2000.0, 1960.0]
        cat = _make_catalogue(ml, year)
        config = GRConfig(mw_min=3.0, mw_max=6.5)

        data = prepare_for_L5(cat, config, UK_COMPLETENESS)
        # 1850 event in window (1900>=1850); 1800 event dropped (1800<1850);
        # buffer 2000 kept (>=1975); buffer 1960 dropped (<1975).
        assert data["n_events"] == 2

        # Opting out reverts to magnitude-only filtering (all 4 above ML threshold).
        data_all = prepare_for_L5(cat, config, UK_COMPLETENESS,
                                  apply_completeness=False)
        assert data_all["n_events"] == 4

    def test_keeps_scatter_in_buffer(self):
        # A buffer event just below mw_min but inside the T_1 window must survive,
        # because the L5 model needs the scatter-IN buffer.
        thresh = ml_threshold(3.0)
        ml = [thresh["ml_threshold"] + 0.01]  # just above detection threshold
        year = [2010.0]
        cat = _make_catalogue(ml, year)
        config = GRConfig(mw_min=3.0, mw_max=6.5)
        data = prepare_for_L5(cat, config, UK_COMPLETENESS)
        assert data["n_events"] == 1
        assert cat.mw[0] < 3.0  # genuinely a sub-mw_min buffer event


class TestBinMagnitudesWindow:
    def test_drops_pre_completeness_events(self):
        # Two Mw~3.7 events (band complete from 1850): keep 1900, drop 1700.
        cat = _make_catalogue([4.0, 4.0], [1900.0, 1700.0])
        binned = bin_magnitudes(cat, mw_min=3.0, mw_max=6.5, model=UK_COMPLETENESS)
        assert int(binned["n"].sum()) == 1
        binned_all = bin_magnitudes(cat, mw_min=3.0, mw_max=6.5,
                                    model=UK_COMPLETENESS, apply_completeness=False)
        assert int(binned_all["n"].sum()) == 2

    def test_no_model_no_filter(self):
        cat = _make_catalogue([4.0, 4.0], [1900.0, 1700.0])
        binned = bin_magnitudes(cat, mw_min=3.0, mw_max=6.5, model=None)
        assert int(binned["n"].sum()) == 2


class TestSyntheticImpact:
    """On synthetic catalogues there are no genuinely-incomplete events, so the
    window filter should only ever *remove* events (never add), and only a small
    boundary fraction — those whose observed magnitude scatters across a
    completeness start-year boundary. It must not silently halve the catalogue."""

    def test_prepare_for_L5_removes_only_small_fraction(self):
        syn = generate_catalogue(UK_COMPLETENESS, b=1.0, lambda_mw_min=5.0,
                                 mw_min=3.0, mw_max=6.5, seed=2024)
        config = GRConfig(mw_min=3.0, mw_max=6.5)
        with_filter = prepare_for_L5(syn["catalogue"], config, UK_COMPLETENESS)
        without = prepare_for_L5(syn["catalogue"], config, UK_COMPLETENESS,
                                 apply_completeness=False)
        assert with_filter["n_events"] <= without["n_events"]
        # boundary scatter only — should be a small fraction, not a halving
        assert with_filter["n_events"] >= 0.8 * without["n_events"]

    def test_bin_magnitudes_removes_only_small_fraction(self):
        syn = generate_catalogue(UK_COMPLETENESS, b=1.0, lambda_mw_min=5.0,
                                 mw_min=3.0, mw_max=6.5, seed=2025)
        a = int(bin_magnitudes(syn["catalogue"], mw_min=3.0, mw_max=6.5,
                               model=UK_COMPLETENESS)["n"].sum())
        b = int(bin_magnitudes(syn["catalogue"], mw_min=3.0, mw_max=6.5,
                               model=UK_COMPLETENESS, apply_completeness=False)["n"].sum())
        assert a <= b
        assert a >= 0.8 * b

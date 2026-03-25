"""Tests for discretization and moment constraint functions."""

import numpy as np
import pytest
from seismicity_analysis.discretization import (
    discretize_marginal, discretize_joint, discretize_five_point,
    build_logic_tree,
    MILLER_RICE, HEAVY_TAIL, ESM, EPT,
)
from seismicity_analysis.types import GRResult
from seismicity_analysis.moment_constraints import (
    seismic_moment, moment_release_rate, moment_release_sensitivity,
)


class TestMarginalDiscretization:
    def test_miller_rice(self):
        d = discretize_marginal(1.0, 0.2, scheme=MILLER_RICE)
        assert len(d["nodes"]) == 3
        assert len(d["weights"]) == 3
        assert sum(d["weights"]) == pytest.approx(1.0)
        # Mean preservation
        assert np.sum(d["nodes"] * d["weights"]) == pytest.approx(1.0, abs=1e-10)
        # Variance preservation
        assert np.sum(d["weights"] * (d["nodes"] - 1.0)**2) == pytest.approx(0.04, abs=1e-10)

    def test_heavy_tail(self):
        d = discretize_marginal(1.0, 0.2, scheme=HEAVY_TAIL)
        assert sum(d["weights"]) == pytest.approx(1.0)
        assert np.sum(d["nodes"] * d["weights"]) == pytest.approx(1.0, abs=1e-10)

    def test_esm(self):
        d = discretize_marginal(1.0, 0.2, scheme=ESM)
        assert sum(d["weights"]) == pytest.approx(1.0)

    def test_ept(self):
        d = discretize_marginal(1.0, 0.2, scheme=EPT)
        assert sum(d["weights"]) == pytest.approx(1.0)


class TestJointDiscretization:
    def test_basic(self):
        mu = [1.0, 1.0]
        s1, s2 = 0.3, 0.2
        rho = -0.5
        Sigma = np.array([[s1**2, rho*s1*s2], [rho*s1*s2, s2**2]])

        d = discretize_joint(mu, Sigma, scheme=HEAVY_TAIL)
        assert len(d["nodes_x1"]) == 9
        assert len(d["nodes_x2"]) == 9
        assert len(d["weights"]) == 9
        assert sum(d["weights"]) == pytest.approx(1.0, abs=1e-10)
        assert np.sum(d["weights"] * d["nodes_x1"]) == pytest.approx(1.0, abs=1e-6)
        assert np.sum(d["weights"] * d["nodes_x2"]) == pytest.approx(1.0, abs=1e-6)


class TestLogicTree:
    def test_from_result(self):
        result = GRResult.create(
            b_mean=1.0, b_sd=0.1,
            lambda_mean=5.0, lambda_sd=1.0,
            rho_lb=-0.5,
            method="test", n_events=100,
        )
        tree = build_logic_tree(result, scheme=HEAVY_TAIL)
        assert len(tree) == 9
        assert sum(br["weight"] for br in tree) == pytest.approx(1.0, abs=1e-10)
        assert all(br["lambda_n"] > 0 for br in tree)


class TestFivePoint:
    def test_basic(self):
        d = discretize_five_point(1.0, 0.2)
        assert len(d["nodes"]) == 5
        assert sum(d["weights"]) == pytest.approx(1.0, abs=1e-3)
        assert np.sum(d["nodes"] * d["weights"]) == pytest.approx(1.0, abs=1e-3)


class TestMomentConstraints:
    def test_basic(self):
        Mdot = moment_release_rate(5.0, 1.0, 3.0, 6.5)
        assert Mdot > 0
        assert np.isfinite(Mdot)

    def test_analytical_vs_numerical(self):
        Mdot_a = moment_release_rate(5.0, 1.0, 3.0, 6.5, method="analytical")
        Mdot_n = moment_release_rate(5.0, 1.0, 3.0, 6.5, method="numerical")
        assert Mdot_a / Mdot_n == pytest.approx(1.0, abs=0.02)

    def test_seismic_moment(self):
        assert seismic_moment(5.0) == pytest.approx(10**(1.5*5 + 9.05))

    def test_monotonicity(self):
        Mdot = moment_release_rate(5.0, 1.0, 3.0, 6.5)
        assert moment_release_rate(10.0, 1.0, 3.0, 6.5) > Mdot  # Higher rate
        assert moment_release_rate(5.0, 1.5, 3.0, 6.5) < Mdot   # Higher b

    def test_sensitivity(self):
        s = moment_release_sensitivity(5.0, 1.0, 3.0, 6.5)
        assert s["eps_lambda"] == pytest.approx(1.0, abs=0.01)
        assert s["eps_b"] < 0
        assert s["eps_Mmax"] > 0

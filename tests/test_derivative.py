"""Tests for numde.derivative (FiniteDerivative)."""

import numpy as np
import pytest

from numde import FiniteDerivative


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Keep x in [0, 1] so function variation within the FD stencil range stays small.
# With step_b = 0.18 and x_max = 2π the function cos(b*x) completes ~1.4 oscillations
# inside the stencil, breaking the polynomial-approximation that FD relies on.
X_ARRAY = np.linspace(0, 1, 300)


def f_sincos(theta, x_array):
    """f(a, b; x) = sin(a*x) * cos(b*x).  Analytic derivatives are known."""
    a, b = theta
    return np.sin(a * x_array) * np.cos(b * x_array)


def _make_nd(N_derivative=2, N_grid=2, a0=2.0, b0=3.0) -> FiniteDerivative:
    nd = FiniteDerivative()
    step_a = a0 * 0.06
    step_b = b0 * 0.06
    nd.define_setup(
        N_derivative=N_derivative,
        N_grid=N_grid,
        expansion_table={"a": [a0, step_a], "b": [b0, step_b]},
    )
    return nd


# ---------------------------------------------------------------------------
# define_setup
# ---------------------------------------------------------------------------


class TestDefineSetup:
    def test_params_stored(self):
        nd = _make_nd()
        assert nd.params == ["a", "b"]
        assert nd.Nparams == 2

    def test_grid_dimensions(self):
        nd = _make_nd(N_grid=3)
        grid = nd._create_parameter_grid("a")
        assert len(grid) == 7  # 2*3 + 1

    def test_grid_centred_on_reference(self):
        nd = _make_nd(a0=5.0)
        grid = nd._create_parameter_grid("a")
        centre = (len(grid) - 1) // 2
        assert grid[centre] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# evaluate_in_grid
# ---------------------------------------------------------------------------


class TestEvaluateInGrid:
    def test_grid_size(self):
        nd = _make_nd(N_grid=2)
        nd.evaluate_in_grid(f_sincos, x_array=X_ARRAY)
        expected = (2 * 2 + 1) ** 2  # 5^2 = 25 grid points for 2 params
        assert len(nd.GRID) == expected

    def test_reference_point_correct(self):
        """Value at index [0, 0] must equal f evaluated at the reference."""
        nd = _make_nd(a0=2.0, b0=3.0)
        nd.evaluate_in_grid(f_sincos, x_array=X_ARRAY)
        ref = nd.GRID["[0, 0]"]
        expected = f_sincos(np.array([2.0, 3.0]), X_ARRAY)
        assert np.allclose(ref, expected)

    def test_caching_no_recompute(self):
        """Serial evaluation must not call the function more times than unique points."""
        call_count = {"n": 0}

        def counting_f(theta, x_array):
            call_count["n"] += 1
            return f_sincos(theta, x_array)

        nd = _make_nd(N_grid=1)
        nd.evaluate_in_grid(counting_f, x_array=X_ARRAY)
        # 9 grid points for 2 params with N_grid=1; all unique → exactly 9 calls
        assert call_count["n"] == 9

    def test_kwargs_forwarded(self):
        """Keyword arguments must reach the function."""
        received = {}

        def probe(theta, multiplier=1.0):
            received["multiplier"] = multiplier
            return np.zeros(5)

        nd = _make_nd(N_grid=1)
        nd.evaluate_in_grid(probe, multiplier=42.0)
        assert received["multiplier"] == 42.0


# ---------------------------------------------------------------------------
# compute_derivatives
# ---------------------------------------------------------------------------


class TestComputeDerivatives:
    @pytest.fixture(scope="class")
    def nd_with_derivatives(self):
        nd = _make_nd(N_derivative=2, N_grid=4)
        nd.evaluate_in_grid(f_sincos, x_array=X_ARRAY)
        nd.compute_derivatives()
        return nd

    def test_expected_labels_present(self, nd_with_derivatives):
        labels = set(nd_with_derivatives.derivatives.keys())
        assert "d1a" in labels
        assert "d1b" in labels
        assert "d2a" in labels
        assert "d2b" in labels
        assert "d1ad1b" in labels

    def test_first_deriv_a_accuracy(self, nd_with_derivatives):
        """∂f/∂a = x*cos(a*x)*cos(b*x); check with the best stencil."""
        nd = nd_with_derivatives
        a0, b0 = nd.expansion_table["a"][0], nd.expansion_table["b"][0]
        best_key = f"{nd.N_grid:02d}"
        computed = nd.derivatives["d1a"][best_key]
        analytic = X_ARRAY * np.cos(a0 * X_ARRAY) * np.cos(b0 * X_ARRAY)
        # Relative tolerance on the RMS
        rms = np.sqrt(np.mean((computed - analytic) ** 2))
        rms_analytic = np.sqrt(np.mean(analytic**2))
        assert rms / rms_analytic < 1e-4

    def test_first_deriv_b_accuracy(self, nd_with_derivatives):
        """∂f/∂b = -x*sin(a*x)*sin(b*x)."""
        nd = nd_with_derivatives
        a0, b0 = nd.expansion_table["a"][0], nd.expansion_table["b"][0]
        best_key = f"{nd.N_grid:02d}"
        computed = nd.derivatives["d1b"][best_key]
        analytic = -X_ARRAY * np.sin(a0 * X_ARRAY) * np.sin(b0 * X_ARRAY)
        rms = np.sqrt(np.mean((computed - analytic) ** 2))
        rms_analytic = np.sqrt(np.mean(analytic**2))
        assert rms / rms_analytic < 1e-4


# ---------------------------------------------------------------------------
# taylor_expand
# ---------------------------------------------------------------------------


class TestTaylorExpand:
    @pytest.fixture(scope="class")
    def nd_ready(self):
        nd = _make_nd(N_derivative=2, N_grid=4)
        nd.evaluate_in_grid(f_sincos, x_array=X_ARRAY)
        nd.compute_derivatives()
        return nd

    def test_at_reference_point(self, nd_ready):
        """Taylor expansion at the reference point should equal f(theta_0)."""
        nd = nd_ready
        a0, b0 = nd.expansion_table["a"][0], nd.expansion_table["b"][0]
        approx = nd.taylor_expand([a0, b0], N_taylor=2)
        exact = f_sincos(np.array([a0, b0]), X_ARRAY)
        assert np.allclose(approx, exact, atol=1e-4)

    def test_nearby_point_accuracy(self, nd_ready):
        """Taylor expansion at a nearby point should be accurate to ~1%."""
        nd = nd_ready
        a0, b0 = nd.expansion_table["a"][0], nd.expansion_table["b"][0]
        a_new, b_new = a0 * 1.02, b0 * 0.98
        approx = nd.taylor_expand([a_new, b_new], N_taylor=2)
        exact = f_sincos(np.array([a_new, b_new]), X_ARRAY)
        rel_rms = np.sqrt(np.mean((approx - exact) ** 2)) / np.sqrt(np.mean(exact**2))
        assert rel_rms < 0.01


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        nd = _make_nd(N_grid=1)
        nd.evaluate_in_grid(f_sincos, x_array=X_ARRAY)
        filepath = str(tmp_path / "grid.npz")
        nd.save_grid_to_file(filepath)

        nd2 = FiniteDerivative()
        nd2.load_grid_from_file(filepath)

        assert nd2.N_grid == nd.N_grid
        assert nd2.params == nd.params
        for key in nd.GRID:
            assert np.allclose(nd2.GRID[key], nd.GRID[key])

    def test_load_nonexistent_raises(self, tmp_path):
        nd = FiniteDerivative()
        with pytest.raises(RuntimeError):
            nd.load_grid_from_file(str(tmp_path / "missing.npz"))

    def test_save_without_grid_raises(self):
        nd = FiniteDerivative()
        with pytest.raises(ValueError, match="No grid"):
            nd.save_grid_to_file("nowhere.npz")

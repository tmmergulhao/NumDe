"""Tests for numde.stencil."""

import numpy as np
import pytest

from numde import StencilCoefficients, apply_stencil_1d


# ---------------------------------------------------------------------------
# StencilCoefficients
# ---------------------------------------------------------------------------


class TestStencilCoefficients:
    """Test the weight computation for various stencils and derivatives."""

    def _weights_dict(self, derivative, stencil, n_params=1):
        """Return {offset_key: weight} from get_coefficients."""
        sc = StencilCoefficients(stencil)
        return {k: v for k, v in sc.get_coefficients(derivative, n_params=n_params)}

    # --- first derivative, 3-point stencil ---

    def test_first_derivative_3pt_weights(self):
        """Central-difference first derivative: c[-1]=-1/2, c[0]=0, c[1]=1/2."""
        w = self._weights_dict([0], stencil=[-1, 0, 1])
        assert w["[-1]"] == pytest.approx(-0.5)
        assert w["[0]"] == pytest.approx(0.0, abs=1e-12)
        assert w["[1]"] == pytest.approx(0.5)

    # --- second derivative, 3-point stencil ---

    def test_second_derivative_3pt_weights(self):
        """Standard 3-point Laplacian: c[-1]=1, c[0]=-2, c[1]=1."""
        w = self._weights_dict([0, 0], stencil=[-1, 0, 1])
        assert w["[-1]"] == pytest.approx(1.0)
        assert w["[0]"] == pytest.approx(-2.0)
        assert w["[1]"] == pytest.approx(1.0)

    # --- second derivative, 5-point stencil ---

    def test_second_derivative_5pt_weights(self):
        """5-point second derivative weights (known values)."""
        w = self._weights_dict([0, 0], stencil=[-2, -1, 0, 1, 2])
        assert w["[-2]"] == pytest.approx(-1 / 12, rel=1e-6)
        assert w["[-1]"] == pytest.approx(4 / 3, rel=1e-6)
        assert w["[0]"] == pytest.approx(-5 / 2, rel=1e-6)
        assert w["[1]"] == pytest.approx(4 / 3, rel=1e-6)
        assert w["[2]"] == pytest.approx(-1 / 12, rel=1e-6)

    # --- weights sum to zero for pure derivatives of order >= 1 ---

    def test_weights_sum_to_zero_first_deriv(self):
        """Weights of any odd-order derivative must sum to zero (symmetry)."""
        sc = StencilCoefficients([-2, -1, 0, 1, 2])
        coeffs = sc.get_coefficients([0], n_params=1)
        total = sum(v for _, v in coeffs)
        assert total == pytest.approx(0.0, abs=1e-12)

    def test_weights_sum_to_zero_second_deriv(self):
        sc = StencilCoefficients([-2, -1, 0, 1, 2])
        coeffs = sc.get_coefficients([0, 0], n_params=1)
        total = sum(v for _, v in coeffs)
        assert total == pytest.approx(0.0, abs=1e-12)

    # --- mixed derivative (two parameters) ---

    def test_mixed_derivative_2params(self):
        """∂²/(∂p0 ∂p1) with 3-point stencil: tensor product of ±1/2."""
        sc = StencilCoefficients([-1, 0, 1])
        coeffs = sc.get_coefficients([0, 1], n_params=2)
        w = {k: v for k, v in coeffs}
        # Only the off-axis corners should be non-zero
        assert w.get("[-1, -1]", 0.0) == pytest.approx(0.25)
        assert w.get("[-1, 1]", 0.0) == pytest.approx(-0.25)
        assert w.get("[1, -1]", 0.0) == pytest.approx(-0.25)
        assert w.get("[1, 1]", 0.0) == pytest.approx(0.25)

    # --- error raised when derivative order exceeds stencil ---

    def test_raises_on_too_high_order(self):
        sc = StencilCoefficients([-1, 0, 1])  # length 3, max order per param = 2
        with pytest.raises(ValueError, match="Stencil of length"):
            sc.get_coefficients([0, 0, 0], n_params=1)  # order 3 not supported

    # --- number of returned points ---

    def test_n_coefficients_single_param(self):
        """For a k-point stencil and 1 param, expect k weights."""
        for k in range(1, 6):
            stencil = np.arange(-k, k + 1)
            sc = StencilCoefficients(stencil)
            coeffs = sc.get_coefficients([0], n_params=1)
            assert len(coeffs) == len(stencil)

    def test_n_coefficients_two_distinct_params(self):
        """Mixed derivative: k^2 combinations for a k-point stencil."""
        sc = StencilCoefficients([-1, 0, 1])
        coeffs = sc.get_coefficients([0, 1], n_params=2)
        assert len(coeffs) == 9  # 3 x 3


# ---------------------------------------------------------------------------
# apply_stencil_1d
# ---------------------------------------------------------------------------


class TestApplyStencil1d:
    """Test the field-array stencil application."""

    def _interior(self, arr, half):
        return arr[half:-half] if half > 0 else arr

    def test_first_derivative_sin(self):
        """d(sin x)/dx ≈ cos x on interior points."""
        N = 500
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        h = x[1] - x[0]
        sc = StencilCoefficients([-2, -1, 0, 1, 2])
        result = apply_stencil_1d(np.sin(x), sc, derivative_order=1, step_size=h)
        half = 2
        assert np.allclose(result[half:-half], np.cos(x)[half:-half], atol=1e-6)

    def test_second_derivative_sin(self):
        """d²(sin x)/dx² ≈ -sin x on interior points."""
        N = 500
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        h = x[1] - x[0]
        sc = StencilCoefficients([-2, -1, 0, 1, 2])
        result = apply_stencil_1d(np.sin(x), sc, derivative_order=2, step_size=h)
        half = 2
        assert np.allclose(result[half:-half], -np.sin(x)[half:-half], atol=1e-5)

    def test_boundary_is_nan(self):
        """Boundary cells must be NaN."""
        x = np.linspace(0, 1, 50)
        h = x[1] - x[0]
        sc = StencilCoefficients([-1, 0, 1])
        result = apply_stencil_1d(x**2, sc, derivative_order=1, step_size=h)
        assert np.isnan(result[0]) and np.isnan(result[-1])

    def test_axis_parameter(self):
        """Differentiating along axis=1 of a 2-D array."""
        # Use 5-point stencil and N=500 so O(h^4) error is well within tolerance
        N = 500
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        field = np.stack([np.sin(x), np.cos(x)])  # shape (2, N)
        h = x[1] - x[0]
        sc = StencilCoefficients([-2, -1, 0, 1, 2])
        result = apply_stencil_1d(field, sc, derivative_order=1, step_size=h, axis=1)
        assert result.shape == field.shape
        half = 2
        # Row 0: d(sin x)/dx ≈ cos x
        assert np.allclose(result[0, half:-half], np.cos(x[half:-half]), atol=1e-6)
        # Row 1: d(cos x)/dx ≈ -sin x
        assert np.allclose(result[1, half:-half], -np.sin(x[half:-half]), atol=1e-6)

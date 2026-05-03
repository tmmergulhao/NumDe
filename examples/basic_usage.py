"""
Basic NumDe usage example.

Demonstrates:
 1. Computing all mixed partial derivatives of a 2-parameter function.
 2. Reconstructing the function at a nearby point via Taylor expansion.
 3. Comparing different stencil accuracies.

No external dependencies beyond NumDe itself.
"""

import numpy as np
import matplotlib.pyplot as plt

from numde import FiniteDerivative, StencilCoefficients, apply_stencil_1d

# ---------------------------------------------------------------------------
# 1. Define the target function
#    f(a, b; x) = sin(a*x) * cos(b*x)
#    Analytic derivatives (at reference a=2, b=3) are known and used for
#    error assessment.
# ---------------------------------------------------------------------------

X = np.linspace(0, 2 * np.pi, 300)

def f(theta, x):
    a, b = theta
    return np.sin(a * x) * np.cos(b * x)

# Reference (expansion) point
A0, B0 = 2.0, 3.0

# ---------------------------------------------------------------------------
# 2. Set up the finite-difference engine
# ---------------------------------------------------------------------------

nd = FiniteDerivative()
nd.define_setup(
    N_derivative=2,
    N_grid=4,          # grid runs from -4 to +4 in each parameter direction
    expansion_table={
        "a": [A0, A0 * 0.06],   # [reference, step_size]
        "b": [B0, B0 * 0.06],
    },
)

# ---------------------------------------------------------------------------
# 3. Evaluate f on the grid (serial, with caching)
# ---------------------------------------------------------------------------

print("Evaluating function on grid ...")
nd.evaluate_in_grid(f, x=X)
print(f"  Grid size: {len(nd.GRID)} points")

# ---------------------------------------------------------------------------
# 4. Compute all partial derivatives up to order 2
# ---------------------------------------------------------------------------

print("Computing derivatives ...")
nd.compute_derivatives()
print(f"  Derivatives computed: {list(nd.derivatives.keys())}")

# ---------------------------------------------------------------------------
# 5. Compare numerical vs analytic first derivatives
# ---------------------------------------------------------------------------

best = f"{nd.N_grid:02d}"   # most accurate stencil key

da_numerical = nd.derivatives["d1a"][best]
da_analytic  = X * np.cos(A0 * X) * np.cos(B0 * X)

db_numerical = nd.derivatives["d1b"][best]
db_analytic  = -X * np.sin(A0 * X) * np.sin(B0 * X)

print(f"\n∂f/∂a  RMS error: {np.sqrt(np.mean((da_numerical - da_analytic)**2)):.2e}")
print(f"∂f/∂b  RMS error: {np.sqrt(np.mean((db_numerical - db_analytic)**2)):.2e}")

# ---------------------------------------------------------------------------
# 6. Taylor expansion at a nearby point
# ---------------------------------------------------------------------------

A_NEW, B_NEW = A0 * 1.03, B0 * 0.97
approx = nd.taylor_expand([A_NEW, B_NEW], N_taylor=2)
exact  = f([A_NEW, B_NEW], X)

rel_rms = np.sqrt(np.mean((approx - exact)**2)) / np.sqrt(np.mean(exact**2))
print(f"\nTaylor approx at (a={A_NEW:.3f}, b={B_NEW:.3f})  relative RMS: {rel_rms:.2e}")

# ---------------------------------------------------------------------------
# 7. Stencil convergence — show how error decreases with stencil size
# ---------------------------------------------------------------------------

print("\nStencil convergence for ∂f/∂a:")
for sk, vals in sorted(nd.derivatives["d1a"].items()):
    rms = np.sqrt(np.mean((vals - da_analytic)**2))
    print(f"  half-width={sk}  RMS={rms:.3e}")

# ---------------------------------------------------------------------------
# 8. apply_stencil_1d demo — spatial derivative on a 1-D array
# ---------------------------------------------------------------------------

print("\napply_stencil_1d demo:")
h = X[1] - X[0]
sc = StencilCoefficients([-2, -1, 0, 1, 2])
d_sin = apply_stencil_1d(np.sin(X), sc, derivative_order=1, step_size=h)
half = 2
err = np.max(np.abs(d_sin[half:-half] - np.cos(X)[half:-half]))
print(f"  max|d(sin x)/dx - cos x| = {err:.2e}  (interior only)")

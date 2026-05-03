"""
NumDe — Multidimensional finite-difference derivatives and Taylor expansions.

Public API
----------
FiniteDerivative
    High-level engine: evaluate a callable on a parameter grid, compute all
    mixed partial derivatives via central finite differences, and reconstruct
    the function at nearby points through a multivariate Taylor expansion.

StencilCoefficients
    Low-level class that computes the weights for any mixed partial derivative
    from a central finite-difference stencil.  Reusable as a building block
    for PDE spatial operators.

apply_stencil_1d
    Apply a :class:`StencilCoefficients` object to a NumPy array along a
    chosen axis.  This is the spatial-derivative primitive for future PDE
    solvers.

Example
-------
>>> import numpy as np
>>> from numde import FiniteDerivative
>>>
>>> x = np.linspace(0, 2 * np.pi, 300)
>>> nd = FiniteDerivative()
>>> nd.define_setup(
...     N_derivative=2,
...     N_grid=2,
...     expansion_table={"a": [2.0, 0.12], "b": [3.0, 0.18]},
... )
>>> nd.evaluate_in_grid(lambda t, x: np.sin(t[0]*x) * np.cos(t[1]*x), x=x)
>>> nd.compute_derivatives()
>>> approx = nd.taylor_expand([2.1, 2.9], N_taylor=2)
"""

from numde.derivative import FiniteDerivative
from numde.stencil import StencilCoefficients, apply_stencil_1d

__all__ = ["FiniteDerivative", "StencilCoefficients", "apply_stencil_1d"]
__version__ = "1.0.0"
__author__ = "Thiago Mergulhao"

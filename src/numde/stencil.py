"""
Finite-difference stencil coefficient computation.

This module provides two public objects:

* :class:`StencilCoefficients` — computes the weights that turn a
  weighted sum of function samples into a numerical derivative of any
  order and any combination of variables.

* :func:`apply_stencil_1d` — applies a :class:`StencilCoefficients`
  object to a NumPy array along a chosen axis.  This is the low-level
  primitive needed by PDE spatial-derivative operators (future
  ``numde.pde`` subpackage) without touching the coefficient math.

Both objects are re-exported from the top-level ``numde`` namespace.
"""

from __future__ import annotations

import itertools
from collections import Counter
from math import factorial

import numpy as np


class StencilCoefficients:
    """Finite-difference coefficients for mixed partial derivatives.

    Given a symmetric integer stencil ``s = [-m, ..., 0, ..., m]``, the
    class builds the associated Vandermonde-style stencil matrix and solves
    the linear system that yields the weights for any partial derivative
    whose per-variable order does not exceed ``len(s) - 1``.

    The coefficients for a *mixed* derivative
    ``∂^(n1+n2+...)/( ∂p0^n1 ∂p1^n2 ...)`` are obtained as the tensor
    product of the independent 1-D coefficient vectors — a standard result
    for central-difference schemes on Cartesian grids.

    Parameters
    ----------
    stencil:
        Integer offsets of the stencil points, e.g. ``[-2, -1, 0, 1, 2]``
        for a 5-point central stencil.  Must be symmetric around zero.

    Examples
    --------
    Second derivative using a 5-point stencil:

    >>> import numpy as np
    >>> from numde import StencilCoefficients
    >>> sc = StencilCoefficients([-2, -1, 0, 1, 2])
    >>> coeffs = sc.get_coefficients([0, 0], n_params=1)
    >>> {k: round(v, 6) for k, v in coeffs}
    {'[-2]': -0.083333, '[-1]': 1.333333, '[0]': -2.5, '[1]': 1.333333, '[2]': -0.083333}
    """

    def __init__(self, stencil: list[int] | np.ndarray) -> None:
        self.stencil: np.ndarray = np.asarray(stencil, dtype=float)
        self.stencil_length: int = len(self.stencil)

        # Row i is stencil ** i  (Vandermonde structure)
        self._matrix: np.ndarray = np.vstack(
            [self.stencil**i for i in range(self.stencil_length)]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_coefficients(
        self,
        derivative: list[int],
        n_params: int,
        verbose: bool = False,
    ) -> list[tuple[str, float]]:
        """Return ``(grid_offset_key, weight)`` pairs for *derivative*.

        Each pair describes one stencil point: the key is the string
        representation of the integer offset vector (length *n_params*)
        and the weight is the corresponding finite-difference coefficient.
        The derivative value is then::

            deriv ≈ sum(weight * f(ref + offset * h) for key, weight in pairs)
                    / prod(h_i for each differentiated variable)

        Parameters
        ----------
        derivative:
            Indices of the parameters to differentiate with respect to,
            with repetition encoding higher-order derivatives.
            ``[0, 0, 1]`` means ``∂³/(∂p0² ∂p1)``.
        n_params:
            Total number of parameters the target function accepts.
        verbose:
            Print each ``(offset_key, weight)`` pair to stdout.

        Returns
        -------
        list of (str, float)
            Stencil point keys and associated weights.

        Raises
        ------
        ValueError
            If any single-variable derivative order equals or exceeds the
            stencil length.
        """
        counts = Counter(derivative)
        orders = list(counts.values())

        if any(order >= self.stencil_length for order in orders):
            raise ValueError(
                f"Stencil of length {self.stencil_length} cannot compute a "
                f"derivative of order {max(orders)}.  "
                f"Minimum stencil length required: {max(orders) + 1}."
            )

        distinct_params = list(counts.keys())
        n_distinct = len(distinct_params)
        half = (self.stencil_length - 1) // 2

        # Solve for 1-D coefficients independently for each parameter
        param_coefs: dict[int, np.ndarray] = {}
        for param_idx in distinct_params:
            rhs = np.zeros(self.stencil_length)
            order = counts[param_idx]
            rhs[order] = float(factorial(order))
            param_coefs[param_idx] = np.linalg.solve(self._matrix, rhs)

        # Tensor-product over all stencil-point combinations
        result: list[tuple[str, float]] = []
        for offsets in itertools.product(range(-half, half + 1), repeat=n_distinct):
            grid_point = np.zeros(n_params, dtype=int)
            weight = 1.0
            for param_idx, offset in zip(distinct_params, offsets):
                grid_point[param_idx] = offset
                weight *= param_coefs[param_idx][offset + half]
            key = str([int(v) for v in grid_point])
            result.append((key, float(weight)))
            if verbose:
                print(key, weight)

        return result


# ---------------------------------------------------------------------------
# PDE building-block: apply stencil to a field array
# ---------------------------------------------------------------------------


def apply_stencil_1d(
    field: np.ndarray,
    stencil: StencilCoefficients,
    derivative_order: int,
    step_size: float,
    axis: int = 0,
) -> np.ndarray:
    """Apply a 1-D finite-difference stencil to *field* along *axis*.

    This function is the low-level primitive for computing spatial
    derivatives on uniform grids — the building block for PDE
    discretisation.  Interior points are computed with the full stencil;
    boundary cells are set to ``numpy.nan`` and must be handled separately
    (e.g. by imposing Dirichlet or Neumann boundary conditions).

    Parameters
    ----------
    field:
        Field values on the uniform grid, shape ``(..., N, ...)``.
    stencil:
        Pre-built :class:`StencilCoefficients` object.
    derivative_order:
        Order of the derivative (e.g. ``1`` for first, ``2`` for second).
    step_size:
        Uniform grid spacing ``h``.
    axis:
        Axis of *field* along which to differentiate.

    Returns
    -------
    numpy.ndarray
        Same shape as *field*.  Boundary cells contain ``numpy.nan``.

    Examples
    --------
    Approximate ``d²sin(x)/dx²`` on a uniform grid:

    >>> import numpy as np
    >>> from numde import StencilCoefficients, apply_stencil_1d
    >>> x = np.linspace(0, 2 * np.pi, 500)
    >>> h = x[1] - x[0]
    >>> sc = StencilCoefficients([-2, -1, 0, 1, 2])
    >>> d2 = apply_stencil_1d(np.sin(x), sc, derivative_order=2, step_size=h)
    >>> np.nanmax(np.abs(d2 + np.sin(x))) < 1e-6   # ≈ -sin(x) inside domain
    True
    """
    import ast

    derivative = [0] * derivative_order
    coefficients = stencil.get_coefficients(derivative, n_params=1)
    half = (stencil.stencil_length - 1) // 2
    n = field.shape[axis]
    n_interior = n - 2 * half

    result = np.full_like(field, np.nan, dtype=float)

    interior = [slice(None)] * field.ndim
    interior[axis] = slice(half, half + n_interior)

    # Accumulate weighted contributions from each stencil offset
    result[tuple(interior)] = 0.0
    for key, weight in coefficients:
        offset = int(ast.literal_eval(key)[0])
        src = [slice(None)] * field.ndim
        src[axis] = slice(half + offset, half + offset + n_interior)
        result[tuple(interior)] = result[tuple(interior)] + weight * field[tuple(src)]

    result[tuple(interior)] /= step_size**derivative_order
    return result

"""
Multidimensional finite-difference derivative engine.

The central class is :class:`FiniteDerivative`.  Given a callable
``f(theta, **kwargs)``, it:

1. Evaluates ``f`` on a regular grid in parameter space centred on a
   reference point.
2. Applies the finite-difference weights from :mod:`numde.stencil` to
   approximate every mixed partial derivative up to a chosen order.
3. Optionally reconstructs ``f`` at any nearby point via a multivariate
   Taylor expansion built from those derivatives.

Grid evaluations and computed derivatives can be persisted to ``.npz``
files so that expensive function calls (e.g. Boltzmann codes) are only
run once.

Typical workflow::

    from numde import FiniteDerivative
    import numpy as np

    nd = FiniteDerivative()
    nd.define_setup(
        N_derivative=2,
        N_grid=2,
        expansion_table={"a": [2.0, 0.12], "b": [3.0, 0.18]},
    )
    nd.evaluate_in_grid(my_function, x_array=np.linspace(0, 1, 200))
    nd.compute_derivatives()
    approx = nd.taylor_expand([2.1, 2.9], N_taylor=2)
"""

from __future__ import annotations

import itertools
from collections import Counter
from functools import partial
from math import factorial
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from numde._utils import derivative_label, multinomial
from numde.stencil import StencilCoefficients


class FiniteDerivative:
    """Compute multidimensional numerical derivatives via finite differences.

    Attributes
    ----------
    parallel : bool
        Use multiprocessing for grid evaluation.  Default ``False``.
    processes : int
        Number of worker processes when ``parallel=True``.  Default ``8``.
    GRID : dict
        Dictionary mapping offset-vector strings to function values after
        :meth:`evaluate_in_grid` is called.
    derivatives : dict
        Nested dictionary of computed derivative arrays after
        :meth:`compute_derivatives` is called.  Outer keys are derivative
        labels (e.g. ``"d2ad1b"``), inner keys are zero-padded stencil
        half-width strings (e.g. ``"02"``).
    """

    def __init__(self) -> None:
        self.parallel: bool = False
        self.processes: int = 8
        self.GRID: dict[str, Any] = {}
        self.derivatives: dict[str, dict[str, Any]] = {}

        # These are set by define_setup or load_grid_from_file
        self.N_derivative: int = 0
        self.N_grid: int = 0
        self.expansion_table: dict[str, list[float]] = {}
        self.params: list[str] = []
        self.Nparams: int = 0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def define_setup(
        self,
        N_derivative: int,
        N_grid: int,
        expansion_table: dict[str, list[float]],
    ) -> None:
        """Configure the derivative engine.

        Parameters
        ----------
        N_derivative:
            Maximum total order of the derivatives to compute.
        N_grid:
            Half-width of the evaluation grid.  The grid runs from
            ``-N_grid`` to ``+N_grid`` along each parameter axis, giving
            ``2*N_grid + 1`` points per dimension.  The largest stencil
            available will have half-width ``N_grid``, giving accuracy of
            order ``O(h^(2*N_grid))``.
        expansion_table:
            Dictionary mapping each parameter name to a two-element list
            ``[reference_value, step_size]``.  For example::

                {"omega_cdm": [0.12, 0.0072], "h": [0.6736, 0.040416]}

        Notes
        -----
        Multiprocessing is controlled by the instance attributes
        ``self.parallel`` (bool) and ``self.processes`` (int); set these
        *after* calling ``define_setup``.
        """
        self.N_derivative = N_derivative
        self.N_grid = N_grid
        self.expansion_table = expansion_table
        self.params = list(expansion_table.keys())
        self.Nparams = len(self.params)

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def _create_parameter_grid(self, param: str) -> np.ndarray:
        """Return the 1-D grid of values for *param*.

        The grid is centred on the reference value and has
        ``2*N_grid + 1`` uniformly spaced points.
        """
        n_stencil = 2 * self.N_grid + 1
        ref, step = self.expansion_table[param]
        offsets = np.arange(n_stencil) - self.N_grid  # [-N, ..., 0, ..., N]
        return ref + step * offsets

    def _generate_all_derivative_cases(
        self, N: int | None = None
    ) -> list[list[list[int]]]:
        """Return all unique partial derivatives up to total order *N*.

        Each element of the outer list corresponds to a total derivative
        order (starting at 1); each inner list is a combination of
        parameter indices (with repetition allowed) that defines one
        partial derivative.

        Parameters
        ----------
        N:
            Maximum total derivative order.  Defaults to ``self.N_derivative``.

        Returns
        -------
        list of list of list of int
            ``output[k]`` contains all derivatives of total order ``k+1``.

        Examples
        --------
        Two parameters, N=2::

            [
                [[0], [1]],                         # order 1
                [[0, 0], [0, 1], [1, 1]],           # order 2
            ]
        """
        max_order = N if N is not None else self.N_derivative
        param_indices = range(self.Nparams)
        return [
            [list(combo) for combo in itertools.combinations_with_replacement(param_indices, k)]
            for k in range(1, max_order + 1)
        ]

    def _normalization(self, derivative: list[int]) -> float:
        """Product of step sizes for the parameters in *derivative*."""
        return float(np.prod([self.expansion_table[self.params[i]][1] for i in derivative]))

    def _derivative_label(self, derivative: list[int]) -> str:
        return derivative_label(derivative, self.params)

    # ------------------------------------------------------------------
    # Grid evaluation
    # ------------------------------------------------------------------

    def evaluate_in_grid(
        self,
        function: Callable,
        save_file: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Evaluate *function* at every point of the parameter grid.

        Parameters
        ----------
        function:
            Callable with signature ``f(theta, **kwargs)`` where *theta*
            is a 1-D NumPy array of parameter values of length
            ``self.Nparams``.  Any extra keyword arguments provided to
            this method are forwarded to *function* on every call.
        save_file:
            If given, persist the grid to this ``.npz`` path after
            evaluation.
        **kwargs:
            Extra arguments forwarded to *function*::

                nd.evaluate_in_grid(f, x_array=np.linspace(0, 1, 200))

        Notes
        -----
        Serial evaluation caches results so that grid points with
        identical parameter values are computed only once.  Set
        ``self.parallel = True`` to use multiprocessing (``Pool.map``);
        in that case, *function* and its keyword arguments must be
        picklable.
        """
        from multiprocessing import Pool

        self.GRID = {}
        total = 2 * self.N_grid + 1

        # All index vectors, e.g. [-1, 0, 1] x [-1, 0, 1] for 2 params, N_grid=1
        index_grid = [
            list(idx)
            for idx in itertools.product(range(-self.N_grid, self.N_grid + 1), repeat=self.Nparams)
        ]

        # Parameter values at each grid point
        grids_1d = np.array([self._create_parameter_grid(p) for p in self.params])
        func_arguments = np.array(
            [
                [grids_1d[j, idx[j] + self.N_grid] for j in range(self.Nparams)]
                for idx in index_grid
            ]
        )

        # Wrap function with fixed keyword arguments
        func = partial(function, **kwargs) if kwargs else function

        results: list[Any] = []

        if self.parallel:
            chunksize = max(1, len(index_grid) // self.processes)
            with Pool(self.processes) as pool:
                results = list(
                    tqdm(pool.map(func, func_arguments, chunksize=chunksize),
                         total=len(func_arguments))
                )
        else:
            cache: dict[bytes, Any] = {}
            for arg in tqdm(func_arguments):
                key = arg.tobytes()
                if key not in cache:
                    cache[key] = func(arg)
                results.append(cache[key])

        for i, idx in enumerate(index_grid):
            self.GRID[str(idx)] = results[i]

        if save_file is not None:
            self.save_grid_to_file(save_file)

    # ------------------------------------------------------------------
    # Derivative computation
    # ------------------------------------------------------------------

    def compute_derivatives(
        self, verbose: bool = False, save_file: str | None = None
    ) -> None:
        """Compute all partial derivatives up to ``self.N_derivative``.

        Results are stored in ``self.derivatives`` as a nested dictionary::

            self.derivatives["d2ad1b"]["02"]  # 2nd-order stencil half-width

        The inner key is a zero-padded integer string equal to the stencil
        half-width.  ``"01"`` is the smallest (3-point) stencil;
        ``"0N"`` with N = ``self.N_grid`` is the most accurate.

        Parameters
        ----------
        verbose:
            Print each derivative label and stencil size as they are computed.
        save_file:
            Existing ``.npz`` file created by :meth:`save_grid_to_file` into
            which the derivatives will be written.
        """
        all_cases = self._generate_all_derivative_cases()
        self.derivatives = {}

        ref_key = str([0] * self.Nparams)

        for order_cases in all_cases:
            for deriv in order_cases:
                label = self._derivative_label(deriv)
                self.derivatives[label] = {}

                if verbose:
                    print(f"{'  ' + label + '  ':*^60}")

                norm = self._normalization(deriv)

                # Iterate over stencils of increasing half-width 1, 2, ..., N_grid.
                # The key is the half-width itself, so "04" means a 9-point stencil
                # spanning -4 to +4.  The most accurate stencil key is f"{N_grid:02d}".
                for half_width in range(1, self.N_grid + 1):
                    stencil_arr = np.arange(-half_width, half_width + 1)
                    stencil_key = f"{half_width:02d}"

                    if verbose:
                        print(f"  stencil half-width: {half_width}")

                    sc = StencilCoefficients(stencil_arr)
                    try:
                        coeffs = sc.get_coefficients(deriv, self.Nparams)
                    except ValueError:
                        continue

                    deriv_value = np.zeros_like(self.GRID[ref_key])
                    for grid_key, weight in coeffs:
                        deriv_value = deriv_value + self.GRID[grid_key] * weight / norm

                    self.derivatives[label][stencil_key] = np.around(deriv_value, 5)

        if save_file is not None:
            data = dict(np.load(save_file, allow_pickle=True).items())
            data["derivatives"] = self.derivatives
            np.savez(save_file, **data)

    # ------------------------------------------------------------------
    # Taylor expansion
    # ------------------------------------------------------------------

    def taylor_expand(
        self,
        x: list[float] | np.ndarray,
        N_taylor: int,
        stencil_key: str | None = None,
    ) -> Any:
        """Approximate ``f(x)`` via a multivariate Taylor expansion.

        Uses the derivatives stored in ``self.derivatives`` to build the
        Taylor series around the reference point defined in
        ``self.expansion_table``.

        Parameters
        ----------
        x:
            Parameter values at which to evaluate the approximation.
            Must have length ``self.Nparams``.
        N_taylor:
            Order of the Taylor expansion (number of derivative orders to
            include).
        stencil_key:
            Which stencil accuracy to use when reading from
            ``self.derivatives``.  Defaults to the most accurate stencil
            (half-width ``N_grid``).

        Returns
        -------
        Any
            Approximated function value (same type/shape as the values in
            ``self.GRID``).
        """
        if stencil_key is None:
            stencil_key = f"{self.N_grid:02d}"
        else:
            stencil_key = str(stencil_key).zfill(2)

        ref_key = str([0] * self.Nparams)
        result = self.GRID[ref_key]

        diffs = [
            float(x[i]) - self.expansion_table[self.params[i]][0]
            for i in range(self.Nparams)
        ]

        for order_idx, order_cases in enumerate(self._generate_all_derivative_cases(N_taylor)):
            order = order_idx + 1
            denominator = factorial(order)
            for deriv in order_cases:
                label = self._derivative_label(deriv)
                counts = Counter(deriv)

                poly = np.prod(
                    [diffs[var] ** power for var, power in counts.items()]
                )
                multi = multinomial(list(counts.values()))

                result = (
                    result
                    + poly * multi * self.derivatives[label][stencil_key] / denominator
                )

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_grid_to_file(self, filename: str) -> None:
        """Persist the evaluation grid to a ``.npz`` file.

        Parameters
        ----------
        filename:
            Output path (will be created or overwritten).

        Raises
        ------
        ValueError
            If no grid has been evaluated yet.
        """
        if not self.GRID:
            raise ValueError("No grid has been computed yet.  Call evaluate_in_grid first.")

        np.savez(
            filename,
            params=self.params,
            GRID=self.GRID,
            N_derivative=self.N_derivative,
            N_grid=self.N_grid,
            expansion_table=self.expansion_table,
        )

    def load_grid_from_file(self, filename: str) -> None:
        """Load a previously saved grid (and optionally derivatives).

        Parameters
        ----------
        filename:
            Path to a ``.npz`` file created by :meth:`save_grid_to_file`.
        """
        try:
            with np.load(filename, allow_pickle=True) as f:
                if "GRID" not in f:
                    raise ValueError("File does not contain a valid GRID array.")
                self.GRID = f["GRID"].item()
                self.params = list(f["params"])
                self.N_derivative = int(f["N_derivative"])
                self.N_grid = int(f["N_grid"])
                self.expansion_table = f["expansion_table"].item()
                self.Nparams = len(self.params)
                if "derivatives" in f.files:
                    self.derivatives = f["derivatives"].item()
        except (IOError, ValueError, KeyError) as exc:
            raise RuntimeError(f"Failed to load grid from '{filename}': {exc}") from exc

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_derivatives(self, x_axis: np.ndarray | None = None) -> None:
        """Plot all computed derivatives, one figure per derivative.

        Parameters
        ----------
        x_axis:
            Optional array to use as the horizontal axis.  If ``None``,
            the sample index is used.
        """
        for label, stencil_dict in self.derivatives.items():
            fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
            ax.set_title(label)
            for sk, values in stencil_dict.items():
                try:
                    if x_axis is not None:
                        ax.plot(x_axis, values, label=sk)
                    else:
                        ax.plot(values, label=sk)
                except (TypeError, ValueError):
                    pass
            ax.legend()
            plt.show()
        plt.close("all")

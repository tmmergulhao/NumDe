# NumDe

A Python package for computing multidimensional numerical derivatives of arbitrary
functions using the central finite-difference method (FDM), and for reconstructing
functions at nearby points via multivariate Taylor expansions.

## Features

- Mixed partial derivatives of any order (e.g. `∂³f / ∂a² ∂b`)
- Automatic computation of all derivatives up to a chosen order
- Stencil sizes from 3-point (2nd-order accurate) to arbitrary precision
- Serial or multiprocessing grid evaluation with transparent caching
- Persist grids and derivatives to `.npz` files for expensive functions
- `apply_stencil_1d` — spatial derivative primitive for future PDE solvers

## Installation

```bash
pip install .
# or for development
pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np
from numde import FiniteDerivative

x = np.linspace(0, 1, 300)

def f(theta, x):
    a, b = theta
    return np.sin(a * x) * np.cos(b * x)

nd = FiniteDerivative()
nd.define_setup(
    N_derivative=2,                          # compute all derivatives up to order 2
    N_grid=4,                                # grid runs from -4 to +4 per parameter
    expansion_table={
        "a": [2.0, 0.12],                    # [reference_value, step_size]
        "b": [3.0, 0.18],
    },
)

nd.evaluate_in_grid(f, x=x)                  # kwargs forwarded to f
nd.compute_derivatives()

# Keys available: d1a, d1b, d2a, d1ad1b, d2b
# Each key maps to a dict of stencil-accuracy levels: "01" (3-pt) … "04" (9-pt)
best_d1a = nd.derivatives["d1a"]["04"]       # ∂f/∂a with 9-point stencil

# Reconstruct f at a nearby point using a 2nd-order Taylor expansion
approx = nd.taylor_expand([2.04, 2.94], N_taylor=2)
```

See `examples/basic_usage.py` for a complete worked example including stencil
convergence plots.  The notebooks in `examples/` demonstrate the cosmology use
case (requires the CLASS Boltzmann code).

## Using `apply_stencil_1d` for spatial derivatives

```python
import numpy as np
from numde import StencilCoefficients, apply_stencil_1d

x = np.linspace(0, 2 * np.pi, 500)
h = x[1] - x[0]

sc = StencilCoefficients([-2, -1, 0, 1, 2])         # 5-point stencil
d2u = apply_stencil_1d(np.sin(x), sc, derivative_order=2, step_size=h)
# d2u ≈ -sin(x) on interior points; boundary cells are NaN
```

## Package layout

```
src/numde/
    __init__.py        public API
    stencil.py         StencilCoefficients, apply_stencil_1d
    derivative.py      FiniteDerivative
    _utils.py          internal helpers
tests/
    test_stencil.py
    test_derivative.py
examples/
    basic_usage.py
    *.ipynb            tutorial notebooks
```

## Running the tests

```bash
pytest
```

## How it works

**Coefficient computation** — for a stencil `s = [-m, …, 0, …, m]`, the class
builds the Vandermonde matrix `M[i,j] = s[j]^i` and solves `M c = e_k` (where
`k` is the derivative order) to obtain the 1-D weights `c`.  For mixed derivatives
the multi-parameter weights are the tensor product of the independent 1-D solutions.

**Grid evaluation** — the function is sampled on a `(2N+1)^p` grid around a
reference point in parameter space (p = number of parameters).  Results are
cached so identical parameter vectors are only evaluated once.

**Taylor expansion** — the stored derivatives are combined with the multinomial
theorem to reconstruct the function at any nearby point up to a specified order.

## License

GNU General Public License v3. See `LICENSE`.

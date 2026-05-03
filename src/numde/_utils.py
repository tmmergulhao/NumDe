"""Internal utility functions for numde."""

from __future__ import annotations

from collections import Counter

from scipy.special import binom


def multinomial(params: list[int]) -> float:
    """Compute the multinomial coefficient for an exponent list.

    Returns ``(n1 + n2 + ... + nk)! / (n1! * n2! * ... * nk!)``, which is
    the combinatorial prefactor that appears in the multivariable Taylor
    expansion of a smooth function.

    Parameters
    ----------
    params:
        Exponents ``[n1, n2, ..., nk]``.

    Returns
    -------
    float
        The multinomial coefficient.
    """
    if len(params) == 1:
        return 1.0
    return float(binom(sum(params), params[-1])) * multinomial(params[:-1])


def derivative_label(derivative: list[int], param_names: list[str]) -> str:
    """Build a human-readable label for a mixed partial derivative.

    Parameters
    ----------
    derivative:
        Parameter indices with repetition encoding derivative order,
        e.g. ``[0, 0, 1]`` means ``d^3 / (dp0^2 dp1)``.
    param_names:
        Ordered list of parameter name strings.

    Returns
    -------
    str
        Label such as ``"d2p0d1p1"``.

    Examples
    --------
    >>> derivative_label([0, 0, 1], ["omega", "h"])
    'd2omegad1h'
    """
    counts = Counter(derivative)
    return "".join(f"d{counts[idx]}{param_names[idx]}" for idx in counts)

"""Aggregation utilities: build and compare pairwise matrices.

These helpers turn an atomic two-item metric (see `pairwise_metrics`) into an
N x N matrix over a collection of items, extract its unique pairwise values, and
regress one such matrix against another.
"""

from typing import Callable, Sequence

import numpy as np
from scipy.stats import linregress, spearmanr

from .sampling_utilities import get_upper_triangle_values

__all__ = [
    'compute_pairwise_matrix',
    'compute_metric_regression',
]


# ==============================================================================
# Build an N x N pairwise matrix from an atomic metric
# ==============================================================================

def compute_pairwise_matrix(
    items: Sequence,
    metric_func: Callable,
    commutative: bool = True,
    skip_diagonal: bool = True,
    diagonal_value: float = np.nan,
) -> np.ndarray:
    """Build an N x N pairwise matrix by applying `metric_func` to every pair.

    `metric_func` must be callable as ``metric_func(a, b)`` (exactly two items).
    If your metric needs extra arguments, bind them ahead of time with
    ``functools.partial`` so it still presents a two-argument interface -- the
    aggregator itself never has to know about those arguments:

        from functools import partial
        from neural_manifold_analysis.pairwise_metrics import rank_order_correlation

        # rank_order_correlation(a, b, circular=False) has a third argument;
        # bind it, then pass the two-argument result:
        metric = partial(rank_order_correlation, circular=True)
        M = compute_pairwise_matrix(items, metric)

    A metric that needs no extra arguments is passed bare:

        M = compute_pairwise_matrix(items, rank_order_correlation)

    Args:
        items: Sequence of N items (e.g. basis matrices, score vectors).
        metric_func: Callable taking two items and returning a scalar. Bind any
            extra arguments with functools.partial first (see above). Example:
            ``partial(rank_order_correlation, circular=True)`` or the bare
            ``procrustes_align`` (returns its disparity by default).
        commutative: If True, assume metric_func(a, b) == metric_func(b, a) and
            compute only the upper triangle, mirroring it. Halves the number of
            evaluations. Set False to evaluate every ordered pair.
        skip_diagonal: If True, do not call metric_func on the diagonal (i, i);
            write `diagonal_value` there instead. Use for self-comparisons that
            are constant by construction (e.g. a distance is 0, a wasted call).
            Set False when the self-value is meaningful (e.g. a similarity of 1).
        diagonal_value: Value written to the diagonal when `skip_diagonal` is
            True. Default np.nan, so the diagonal is excluded by
            `get_upper_triangle_values` / NaN-aware reductions downstream.

    Returns:
        (N, N) matrix of pairwise metric values.
    """
    n = len(items)
    matrix = np.full((n, n), np.nan)

    for i in range(n):
        # Diagonal
        if not skip_diagonal:
            matrix[i, i] = metric_func(items[i], items[i])
        else:
            matrix[i, i] = diagonal_value

        if commutative:
            # Upper triangle only, then mirror.
            for j in range(i + 1, n):
                v = metric_func(items[i], items[j])
                matrix[i, j] = v
                matrix[j, i] = v
        else:
            for j in range(n):
                if i == j:
                    continue
                matrix[i, j] = metric_func(items[i], items[j])

    return matrix


# ==============================================================================
# Regress one pairwise matrix against another
# ==============================================================================

def compute_metric_regression(matrix_x: np.ndarray, matrix_y: np.ndarray) -> dict:
    """OLS regression and Spearman correlation between two pairwise matrices.

    Extracts the upper-triangle values of both matrices, drops pairs where
    either value is NaN, and relates the two.

    Args:
        matrix_x: Square (N, N) pairwise matrix (independent variable).
        matrix_y: Square (N, N) pairwise matrix (dependent variable).

    Returns:
        Dict with keys: slope, intercept, r_value, p_value, stderr,
        spearman_rho, spearman_p.
    """
    x = get_upper_triangle_values(matrix_x)
    y = get_upper_triangle_values(matrix_y)
    valid = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[valid], y[valid]

    slope, intercept, r_value, p_value, stderr = linregress(x, y)
    rho, p_rho = spearmanr(x, y)

    return {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'stderr': stderr,
        'spearman_rho': rho,
        'spearman_p': p_rho,
    }
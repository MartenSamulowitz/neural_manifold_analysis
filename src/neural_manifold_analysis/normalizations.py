"""General-purpose centering, standardization, and normalization utilities.

Two families of functions live here:

* Per-axis / whole-array utilities (`center`, `standardize`, `l1_normalize`,
  `l2_normalize`) that operate on the rows (axis=1), columns (axis=0), or the
  entire array (axis=None) of any 2-D matrix.
* Symmetric-matrix utilities (`normalize_upper_triangle`) that normalize a
  pairwise symmetric matrix using only its off-diagonal values and return a
  symmetric square matrix with the diagonal preserved.

All functions are NaN-aware: NaNs are ignored when computing statistics and
NaN entries are passed through to the output unchanged.
"""

from typing import Callable

import numpy as np

__all__ = [
    'center',
    'standardize',
    'l1_normalize',
    'l2_normalize',
    'normalize_upper_triangle',
]


# ==============================================================================
# Centering
# ==============================================================================

def center(data: np.ndarray, axis: int | None = 1) -> np.ndarray:
    """Subtract the mean along `axis` so each slice becomes zero-mean.

    NaN-aware: the mean ignores NaNs (np.nanmean); NaN entries are passed
    through unchanged.

    Idempotent: if the data is already zero-mean (max absolute mean < 1e-12),
    the input is returned unchanged to avoid an unnecessary allocation.

    Args:
        data: Input array.
        axis: Axis to center along -- 1 per row, 0 per column, None over the
            whole array (default 1).

    Returns:
        Centered array of the same shape.
    """
    means = np.nanmean(data, axis=axis, keepdims=True)
    if np.abs(means).max() < 1e-12:
        return data
    return data - means


# ==============================================================================
# Standardization
# ==============================================================================

def standardize(data: np.ndarray, axis: int | None = 1) -> np.ndarray:
    """Z-score along `axis`: subtract the mean and divide by the std.

    NaN-aware: mean and std ignore NaNs (np.nanmean / np.nanstd); NaN entries
    are passed through unchanged. Slices with zero standard deviation are
    divided by 1.0, so they become all-zero after centering rather than NaN.

    Args:
        data: Input array.
        axis: Axis to standardize along -- 1 per row, 0 per column, None over
            the whole array (default 1).

    Returns:
        Standardized array of the same shape.
    """
    centered = center(data, axis=axis)
    stds = np.nanstd(centered, axis=axis, ddof=0, keepdims=True)
    stds_safe = np.where(stds == 0, 1.0, stds)
    return centered / stds_safe


# ==============================================================================
# L1-normalization
# ==============================================================================

def l1_normalize(data: np.ndarray, axis: int | None = 1) -> np.ndarray:
    """Scale each slice along `axis` to unit L1 norm (sum of absolute values).

    Divides every slice by its L1 norm ``sum(|x|)``. The mean is left intact --
    the raw vector is normalized, not the centered one.

    NaN-aware: NaNs are ignored when computing the norm (np.nansum); NaN entries
    are passed through unchanged. Slices with zero norm are divided by 1.0 and
    returned unchanged (all-zero slices stay all-zero).

    Compared with the other normalizers:
      - l1_normalize -- divides by sum(|x|); robust to large outliers, tends to
                        preserve sparsity.
      - l2_normalize -- divides by sqrt(sum(x^2)); the Euclidean unit-norm.
      - standardize  -- centers AND divides by std; unlike these, it removes the
                        mean and can inflate the noise of flat slices.
    Pass axis=None to normalize by the norm of the whole array (Frobenius-style,
    a single scalar that preserves relative scale between slices).

    Args:
        data: Input array.
        axis: Axis to normalize along -- 1 per row, 0 per column, None over the
            whole array (default 1).

    Returns:
        Array of the same shape with unit-L1-norm slices along `axis`.
    """
    norms = np.nansum(np.abs(data), axis=axis, keepdims=True)
    norms_safe = np.where(norms == 0, 1.0, norms)
    return data / norms_safe


# ==============================================================================
# L2-normalization
# ==============================================================================

def l2_normalize(data: np.ndarray, axis: int | None = 1) -> np.ndarray:
    """Scale each slice along `axis` to unit L2 (Euclidean) norm.

    Divides every slice by its L2 norm ``sqrt(sum(x^2))``. The mean is left
    intact -- the raw vector is normalized, not the centered one.

    NaN-aware: NaNs are ignored when computing the norm (np.nansum); NaN entries
    are passed through unchanged. Slices with zero norm are divided by 1.0 and
    returned unchanged (all-zero slices stay all-zero).

    Compared with the other normalizers:
      - l2_normalize -- divides by sqrt(sum(x^2)); the Euclidean unit-norm.
      - l1_normalize -- divides by sum(|x|); more robust to large outliers.
      - standardize  -- centers AND divides by std; unlike these, it removes the
                        mean and can inflate the noise of flat slices.
    Pass axis=None for a whole-array (Frobenius) norm: a single scalar that
    preserves relative scale between slices and keeps the input shape.

    Args:
        data: Input array.
        axis: Axis to normalize along -- 1 per row, 0 per column, None over the
            whole array (default 1).

    Returns:
        Array of the same shape with unit-L2-norm slices along `axis`.
    """
    norms = np.sqrt(np.nansum(data**2, axis=axis, keepdims=True))
    norms_safe = np.where(norms == 0, 1.0, norms)
    return data / norms_safe


# ==============================================================================
# Symmetric-matrix normalization (off-diagonal only)
# ==============================================================================

def normalize_upper_triangle(
    matrix: np.ndarray,
    func: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Normalize a symmetric matrix using only its off-diagonal values.

    A pairwise symmetric matrix cannot be normalized per row or column without
    breaking symmetry, and its diagonal is usually structural (e.g. all ones for
    correlations, all zeros for distances) and would contaminate statistics
    computed over the whole matrix. This function therefore:

      1. Extracts the strictly upper-triangular values (k=1) -- each unordered
         pair appears exactly once.
      2. Applies `func` to the valid (non-NaN) values only.
      3. Writes the transformed values back symmetrically (out[i, j] == out[j, i]).
      4. Leaves NaN pairs as NaN and copies the original diagonal through.

    Args:
        matrix: Square symmetric matrix of shape (N, N).
        func: Callable mapping a 1-D array of the valid off-diagonal values to a
            1-D array of the same length (e.g. a normalization or z-score). It
            only ever receives the non-NaN entries, so it need not handle NaNs.
            To reuse the per-axis normalizers here, reduce over the whole vector
            with axis=None, e.g. ``lambda v: l2_normalize(v, axis=None)``.

    Returns:
        Symmetric matrix of the same shape: off-diagonal entries hold the
        transformed values (NaN where the input pair was NaN); the diagonal is
        copied unchanged from the input.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square 2-D matrix, got shape {matrix.shape}.")

    idx_i, idx_j = np.triu_indices(matrix.shape[0], k=1)
    vals = matrix[idx_i, idx_j]
    valid = ~np.isnan(vals)

    transformed = vals.copy()
    transformed[valid] = func(vals[valid])

    out = np.full_like(matrix, np.nan)
    out[idx_i, idx_j] = transformed
    out[idx_j, idx_i] = transformed
    np.fill_diagonal(out, np.diag(matrix))
    return out

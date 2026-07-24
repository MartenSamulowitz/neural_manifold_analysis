"""Sampling and array-subset utilities.

Small helpers for drawing random subsets of an array and for extracting the
unique (upper-triangle) entries of a square matrix.
"""

import numpy as np

__all__ = [
    'get_upper_triangle_values',
    'subsample',
    'shuffle',
    'resample_bins',
]


# ==============================================================================
# Extract the upper triangle of a square matrix
# ==============================================================================

def get_upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    """Extract the strict upper triangle of a square matrix as a flat 1-D array.

    Returns only the above-diagonal entries, i.e. each unordered pair exactly
    once. This strips the redundant mirrored half out of a symmetric matrix.

    Args:
        matrix: Square array of shape (n, n).

    Returns:
        1-D array of length n*(n-1)/2 holding the upper-triangle values.
    """
    mask = np.triu(np.ones(matrix.shape, dtype=bool), k=1)
    return matrix[mask]


# ==============================================================================
# Random subsampling along an axis
# ==============================================================================

def subsample(
    data: np.ndarray,
    fraction: float,
    axis: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a random subset of `data` along the given axis (without replacement).

    Args:
        data: Input array.
        fraction: Fraction of entries along `axis` to keep, in (0, 1]. A value of
            1.0 returns a copy of the full array.
        axis: Axis to subsample along.
        rng: NumPy random Generator. If None, a fresh default_rng() is used.

    Returns:
        Array with the same number of dimensions as `data`, reduced to
        floor(fraction * size) entries along `axis`.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}.")
    if axis < 0 or axis >= data.ndim:
        raise ValueError(f"axis must be in [0, {data.ndim - 1}], got {axis}.")

    if fraction == 1:
        return data.copy()

    if rng is None:
        rng = np.random.default_rng()

    size = data.shape[axis]
    num_to_sample = int(size * fraction)
    indices = rng.choice(size, size=num_to_sample, replace=False)
    return np.take(data, indices, axis=axis)


# ==============================================================================
# Shuffle values along an axis
# ==============================================================================

def shuffle(
    data: np.ndarray,
    axis: int | None = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a copy of `data` with its values randomly permuted.

    Each 1-D slice along `axis` is permuted independently, so structure across
    that axis is destroyed while the slices stay aligned. If `axis` is None, all
    values are shuffled together across the flattened array.

    Args:
        data: Input array of any dimensionality.
        axis: Axis to shuffle along (default 1). If None, shuffle all values
            globally across the whole array.
        rng: NumPy random Generator. If None, a fresh default_rng() is used.

    Returns:
        A new array of the same shape with permuted values.
    """
    if rng is None:
        rng = np.random.default_rng()

    if axis is None:
        return rng.permutation(data.ravel()).reshape(data.shape)

    if axis < 0 or axis >= data.ndim:
        raise ValueError(f"axis must be in [0, {data.ndim - 1}] or None, got {axis}.")

    # Independent permutation per 1-D slice: argsort of random keys along `axis`.
    order = np.argsort(rng.random(data.shape), axis=axis)
    return np.take_along_axis(data, order, axis=axis)


# ==============================================================================
# Resample
# ==============================================================================

def resample_bins(values: np.ndarray, n_target: int, intensive: bool = True) -> np.ndarray:
    """Re-bin ``values`` along its last axis to ``n_target`` bins via CDF interpolation.

    Splits or merges bins by linearly interpolating the cumulative sum at the new
    bin edges, apportioning each source bin exactly (the only assumption is uniform
    density within a source bin).

    Parameters
    ----------
    values
        ``(..., n_source)`` array; the last axis holds the bins to resample
        (e.g. spatial or temporal bins of a rate or count signal).
    n_target
        Desired number of bins after re-binning.
    intensive
        If ``True`` (default) values are treated as an intensive density (e.g. a
        rate) and rescaled by ``n_target / n_source`` so magnitude is
        resolution-invariant. Set ``False`` for an extensive quantity (e.g. raw
        counts), where total mass is conserved without rescaling.

    Returns
    -------
    ``(..., n_target)`` re-binned array.
    """
    values = np.asarray(values, dtype=float)
    n_source = values.shape[-1]
    if n_source == n_target:
        return values.copy()

    # Old and new bin edges, both tiling the same unit interval [0, 1].
    old_edges = np.linspace(0.0, 1.0, n_source + 1)
    new_edges = np.linspace(0.0, 1.0, n_target + 1)

    # CDF at old edges: leading 0, then cumulative sum along the bin axis.
    lead = np.zeros(values.shape[:-1] + (1,))
    cdf = np.concatenate([lead, np.cumsum(values, axis=-1)], axis=-1)

    # Linear-interpolate the CDF at the new edges (same weights for every row,
    # so we build them once and apply along the last axis).
    idx = np.searchsorted(old_edges, new_edges, side="right") - 1
    idx = np.clip(idx, 0, n_source - 1)
    left = old_edges[idx]
    width = old_edges[idx + 1] - left
    frac = (new_edges - left) / width                      # 0..1 within old bin
    cdf_new = cdf[..., idx] * (1.0 - frac) + cdf[..., idx + 1] * frac

    out = np.diff(cdf_new, axis=-1)                         # mass per new bin
    if intensive:
        out *= n_target / n_source                         # mass -> rate density
    return out

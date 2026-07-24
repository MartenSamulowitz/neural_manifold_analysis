"""Atomic pairwise metrics between two items.

Each function measures the relationship between exactly two inputs and returns a
scalar similarity or distance, so any of them can be passed directly to the
aggregators in `aggregation_utilities` (e.g. `compute_pairwise_matrix`). The
metrics are commutative, f(A, B) == f(B, A), so they can drive
`compute_pairwise_matrix(..., commutative=True)`.

(`procrustes_align` returns its disparity by default; pass `return_aligned=True`
to also get the two aligned matrices back.)
"""

import numpy as np
from scipy.spatial import procrustes
from scipy.stats import spearmanr

__all__ = [
    'rank_order_correlation',
    'jaccard_similarity',
    'procrustes_align',
]


# ==============================================================================
# Rank-order correlation between two score vectors
# ==============================================================================

def rank_order_correlation(data_A: np.ndarray, data_B: np.ndarray, circular: bool = False) -> float:
    """Rank-order similarity between two score vectors.

    circular=False : Spearman correlation in [-1, 1]. Sensitive to absolute rank
        position (a constant shift of all ranks lowers the score).
    circular=True  : circular rank similarity in [0, 1], invariant to circular
        shift AND reversal. Use when the ordering lives on a ring and its
        orientation is arbitrary. ~1/sqrt(N) is the chance level for two
        independent random orderings.

    Args:
        data_A, data_B: 1-D score vectors of equal length.
        circular: Select the circular variant (see above).

    Returns:
        Scalar similarity (range depends on `circular`).
    """
    if not circular:
        return float(spearmanr(data_A, data_B).correlation)

    N = len(data_A)
    # Double argsort -> circular rank (position) of each element.
    rank_a = np.argsort(np.argsort(data_A)).astype(float)
    rank_b = np.argsort(np.argsort(data_B)).astype(float)
    z_a = np.exp(2j * np.pi * rank_a / N)
    z_b = np.exp(2j * np.pi * rank_b / N)
    forward = np.abs(np.sum(np.conj(z_a) * z_b)) / N  # shift-invariant
    reverse = np.abs(np.sum(z_a * z_b)) / N           # shift + reversal invariant
    return float(max(forward, reverse))


# ==============================================================================
# Jaccard similarity between two binary vectors
# ==============================================================================

def jaccard_similarity(data_A: np.ndarray, data_B: np.ndarray) -> float:
    """Jaccard similarity between two binary (or truthy) vectors.

    Jaccard = |A intersection B| / |A union B|, in [0, 1]. Two empty vectors
    (empty union) are defined as perfectly similar and return 1.0.

    Args:
        data_A, data_B: Equal-length vectors interpreted as boolean masks.

    Returns:
        Scalar similarity in [0, 1].
    """
    if len(data_A) != len(data_B):
        raise ValueError(f"Both inputs must have the same length; got {len(data_A)} vs {len(data_B)}")
    
    intersection = np.sum(np.logical_and(data_A, data_B))
    union = np.sum(np.logical_or(data_A, data_B))
    if union == 0:
        return 1.0  # both vectors empty -> perfect similarity
    return intersection / union


# ==============================================================================
# Procrustes alignment between two datasets
# ==============================================================================

def procrustes_align(
    data1: np.ndarray,
    data2: np.ndarray,
    return_aligned: bool = False,
) -> float | tuple[np.ndarray, np.ndarray, float]:
    """Align two datasets by Procrustes analysis and report their shape disparity.

    Compares the intrinsic geometry (shape, up to rotation/scale/translation) of
    two datasets. Alignment only: any standardization, dimensionality reduction,
    or truncation must be applied beforehand.

    By default returns just the scalar disparity, so it can be passed directly as
    a metric to `compute_pairwise_matrix`. Set `return_aligned=True` to also get
    the two aligned matrices back.

    Args:
        data1, data2: Arrays of identical shape (e.g. (n_observations, n_components)).
        return_aligned: If True, also return the two aligned matrices (see Returns).

    Returns:
        If `return_aligned` is False (default): ``disparity`` — the squared
            residual after optimal alignment (0 == identical shape).
        If `return_aligned` is True: ``(mtx1, mtx2, disparity)``, where ``mtx1`` is
            standardized `data1` and ``mtx2`` is `data2` aligned to ``mtx1``.
    """
    if data1.shape != data2.shape:
        raise ValueError(
            f"Both inputs must have the same shape; got {data1.shape} vs {data2.shape}"
        )
    mtx1, mtx2, disparity = procrustes(data1, data2)
    if return_aligned:
        return mtx1, mtx2, disparity
    return disparity

"""PCA helpers: fit a PCA model, sort features by loading angle, and extract bases.

All functions operate on 2-D arrays of shape (n_features, n_samples): rows are
features, columns are samples.
"""

import numpy as np
from sklearn import decomposition
from .normalizations import center

__all__ = [
    'compute_pca', 
    'pca_angle_sorting',
    'pca_extract_bases',
]


# ==============================================================================
# PCA
# ==============================================================================

def compute_pca(data: np.ndarray, num_components: int) -> decomposition.PCA:
    """Fit PCA on the data, treating rows as features and columns as samples.

    The caller is responsible for transposing the data if necessary. 

    Args:
        data: Array of shape (n_features, n_samples).
        num_components: Number of principal components to retain.

    Returns:
        Fitted sklearn PCA object. Key attributes:
        - .components_: (num_components, n_features) — PC loading vectors
        - .explained_variance_ratio_: fraction of variance per component
        - .transform(data.T).T: project data onto PC space → (num_components, n_samples)
    """
    data = center(data, axis=1)
    pca = decomposition.PCA(n_components=num_components)
    pca.fit(data.T)
    return pca


# ==============================================================================
# Sort features by angle in PC1-PC2 loading space
# ==============================================================================

def pca_angle_sorting(data: np.ndarray) -> np.ndarray:
    """Sort features by their angle in PC1-PC2 loading space (ascending, mod 2*pi).

    Args:
        data: Array of shape (n_features, n_samples).

    Returns:
        1-D array of indices that orders the features by loading angle.
    """
    pca = compute_pca(data, num_components=2)

    loadings = pca.components_[:2, :]
    angles = np.arctan2(loadings[1, :], loadings[0, :])

    order = np.argsort(angles % (2 * np.pi))
    return order


# ==============================================================================
# Extract the top-k orthonormal basis from a PCA object
# ==============================================================================

def pca_extract_bases(pca_object: decomposition.PCA, k: int) -> np.ndarray:
    """Extract the top-k orthonormal basis from a fitted PCA object.

    Returns the first k principal directions as column vectors, so the result
    can be compared (subspace angles, projections) without carrying the full
    PCA object.

    Args:
        pca_object: A fitted sklearn PCA object. Its `.components_` has shape
            (num_components, num_features).
        k: Number of leading principal components to keep.

    Returns:
        Orthonormal basis of shape (num_features, k); columns are the top-k
        principal directions (unit-norm, mutually orthogonal).
    """
    n_components = pca_object.components_.shape[0]
    if k > n_components:
        raise ValueError(
            f"Requested k={k}, but the PCA object only has {n_components} components."
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    return pca_object.components_[:k, :].T




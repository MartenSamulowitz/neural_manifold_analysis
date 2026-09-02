"""Dimensionality estimation.

All functions operate on 2-D arrays of shape (n_features, n_samples): rows are
features, columns are samples.
"""

import numpy as np
from scipy import stats

from .normalizations import center
from .pca import compute_pca

__all__ = [
    'threshold_dim',
    'participation_ratio',
    'evaluate_eigenspectrum_against_null',
    'parallel_analysis',
    'parallel_analysis_fast',
]


#==============================================================================
# Dimensionality Estimation via Thresholding
# ==============================================================================

def threshold_dim(
    data: np.ndarray,
    threshold: float = 0.9,
) -> int:
    """Estimate dimensionality as the number of PCA components needed to reach a
    cumulative variance threshold.

    ``compute_pca`` mean-centers each row before fitting, so no explicit centering
    is done here. No variance normalisation is applied — the input is expected to
    be globally z-scored upstream.

    Args:
        data: Array of shape (n_features, n_samples).
        threshold: Cumulative explained-variance ratio to reach (default 0.9).

    Returns:
        Number of components required to explain at least ``threshold`` variance.
    """
    try:
        pca = compute_pca(data, num_components=min(data.shape))
        eigenvalues = pca.explained_variance_
    except np.linalg.LinAlgError:
        # SVD may fail on very short inputs; fall back to eigvalsh on covariance
        centered_data = center(data)
        n, t = centered_data.shape
        if n <= t:
            cov = centered_data @ centered_data.T / (t - 1)
        else:
            cov = centered_data.T @ centered_data / (t - 1)
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    explained_variance_ratio = eigenvalues / eigenvalues.sum()
    cumulative_variance = np.cumsum(explained_variance_ratio)
    num_components = np.argmax(cumulative_variance >= threshold) + 1
    return num_components


# ==============================================================================
# Dimensionality Estimation via Participation Ratio
# ==============================================================================

def participation_ratio(data: np.ndarray) -> float:
    """Calculate the Participation Ratio (PR), an effective-dimensionality measure.

    PR is defined as ``(sum of eigenvalues)^2 / sum of squared eigenvalues``.
    ``compute_pca`` mean-centers each row before fitting, so no explicit centering
    is done here. No variance normalisation is applied — the input is expected to
    be globally z-scored upstream.

    Args:
        data: Array of shape (n_features, n_samples).

    Returns:
        Participation ratio of the data covariance.
    """
    try:
        pca = compute_pca(data, num_components=min(data.shape))
        eigenvalues = pca.explained_variance_
    except np.linalg.LinAlgError:
        centered_data = center(data)
        n, t = centered_data.shape
        if n <= t:
            cov = centered_data @ centered_data.T / (t - 1)
        else:
            cov = centered_data.T @ centered_data / (t - 1)
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    pr = (eigenvalues.sum() ** 2) / np.sum(eigenvalues ** 2)
    return pr


# ==============================================================================
# Eigenspectrum Evaluation (shared decision rule used by both PA functions)
# ==============================================================================

def evaluate_eigenspectrum_against_null(
    real_eigenvalues: np.ndarray,
    null_mean: np.ndarray,
    null_std: np.ndarray,
    confidence_level: float = 0.999,
) -> int:
    """Count eigenvalues confidently above a null distribution via a two-tailed z-test.

    This is the single canonical implementation of the PA decision rule. Both
    parallel_analysis() and parallel_analysis_fast() call this function at each
    convergence check, guaranteeing that cross-run comparisons (e.g. applying a
    null estimated on one dataset to the eigenspectrum of a subset) use identical logic.

    Args:
        real_eigenvalues: Shape (K,) — real eigenvalues in descending order.
        null_mean:        Shape (K,) — per-component null mean from a PA run on the
                          same recording (same N, hence same K).
        null_std:         Shape (K,) — per-component null std from the same PA run.
        confidence_level: Two-tailed confidence (default 0.999, z_crit ≈ 3.29).

    Returns:
        int: number of components confidently above the null — the dimensionality estimate.

    Raises:
        ValueError: if real_eigenvalues and null_mean have different lengths, which
                    would indicate mismatched recordings (different N).
    """
    if len(real_eigenvalues) != len(null_mean) or len(real_eigenvalues) != len(null_std):
        raise ValueError(
            f"Array length mismatch: real_eigenvalues ({len(real_eigenvalues)}), "
            f"null_mean ({len(null_mean)}), null_std ({len(null_std)}). "
            "All must have the same length (num_components = min(N, T)). "
            "Mismatched lengths indicate eigenspectra from different recordings were mixed."
        )

    alpha = 1.0 - confidence_level
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

    n_std = np.maximum(null_std, 1e-15)
    z = (real_eigenvalues - null_mean) / n_std

    is_above = z > z_crit

    # Tail protection: components where both real and null are negligibly small
    # are unconditionally classified as noise regardless of z-score.
    is_tail  = (real_eigenvalues < 1e-10) & (null_mean < 1e-10)
    is_above = is_above & ~is_tail

    return int(np.sum(is_above))


# ==============================================================================
# Dimensionality Estimation via Parallel Analysis
# ==============================================================================

def parallel_analysis(
    data: np.ndarray,
    max_shuffles: int = 100,
    min_shuffles: int = 10,
    confidence_level: float = 0.999,
    rng: np.random.Generator | None = None,
    return_spectrum: bool = False,
) -> tuple[int, int] | tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Parallel Analysis with STD-based z-test and patience-based early stopping.

    Uses evaluate_eigenspectrum_against_null() for all dimensionality decisions,
    guaranteeing that cross-run null comparisons use the identical decision rule.
    Data is mean-centered per neuron (row) before PCA — no variance normalisation applied.

    Returns:
        If return_spectrum is False: tuple (estimated_dimensionality, shuffles_used)
        If return_spectrum is True:  tuple (estimated_dimensionality, shuffles_used,
            real_eigenvalues, null_mean, null_std) — all shape (num_components,)
    """
    if rng is None:
        rng = np.random.default_rng()

    num_neurons, num_frames = data.shape
    num_components = min(num_neurons, num_frames)

    centered_data = center(data)
    real_eigenvalues = compute_pca(
        centered_data, num_components=num_components
    ).explained_variance_

    null_eigenvalues_history = np.zeros((max_shuffles, num_components), dtype=float)

    patience = min_shuffles
    prev_dim = -1
    stable_count = 0

    for current_shuffle in range(max_shuffles):
        # 1. Shuffle each neuron's time series independently (destroys correlations)
        random_indices = rng.random((num_neurons, num_frames)).argsort(axis=1)
        shuffled_data = np.take_along_axis(centered_data, random_indices, axis=1)

        # 2. Null eigenvalues via full PCA
        null_eigenvalues_history[current_shuffle, :] = compute_pca(
            shuffled_data, num_components=num_components
        ).explained_variance_

        number_of_completed_shuffles = current_shuffle + 1

        # 3. Wait for minimum sample size before any stopping decision
        if number_of_completed_shuffles < min_shuffles:
            continue

        # 4. Estimate null distribution parameters
        current_null_runs = null_eigenvalues_history[:number_of_completed_shuffles]
        null_mean = current_null_runs.mean(axis=0)
        null_std  = np.maximum(current_null_runs.std(axis=0, ddof=1), 1e-15)

        # 5. Evaluate using the shared decision rule
        current_dim = evaluate_eigenspectrum_against_null(
            real_eigenvalues, null_mean, null_std, confidence_level
        )

        # 6. Check for full convergence: are all components resolved?
        n_std_clipped = np.maximum(null_std, 1e-15)
        z = (real_eigenvalues - null_mean) / n_std_clipped
        is_tail = (real_eigenvalues < 1e-10) & (null_mean < 1e-10)
        is_uncertain = (np.abs(z) <= stats.norm.ppf(1.0 - (1.0 - confidence_level) / 2.0)) & ~is_tail
        if not np.any(is_uncertain):
            if return_spectrum:
                return current_dim, number_of_completed_shuffles, real_eigenvalues, null_mean, null_std
            return current_dim, number_of_completed_shuffles

        # 7. Patience-based stopping: dim stable for `patience` consecutive shuffles
        if current_dim == prev_dim:
            stable_count += 1
        else:
            stable_count = 0
            prev_dim = current_dim
        if stable_count >= patience:
            if return_spectrum:
                return current_dim, number_of_completed_shuffles, real_eigenvalues, null_mean, null_std
            return current_dim, number_of_completed_shuffles

    null_mean = null_eigenvalues_history.mean(axis=0)
    null_std  = np.maximum(null_eigenvalues_history.std(axis=0, ddof=1), 1e-15)
    dim = evaluate_eigenspectrum_against_null(real_eigenvalues, null_mean, null_std, confidence_level)
    if return_spectrum:
        return dim, max_shuffles, real_eigenvalues, null_mean, null_std
    return dim, max_shuffles


# ==============================================================================
# Fast Parallel Analysis (eigenvalue-only, for mass pairwise application)
# ==============================================================================

def parallel_analysis_fast(
    data: np.ndarray,
    max_shuffles: int = 100,
    min_shuffles: int = 10,
    confidence_level: float = 0.999,
    rng: np.random.Generator | None = None,
    return_spectrum: bool = False,
) -> tuple[int, int] | tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Fast parallel analysis using eigvalsh on the covariance matrix.

    Functionally identical to parallel_analysis(), but skips eigenvector
    computation (no sklearn PCA object, no SVD) for speed in mass-application
    settings (e.g. estimating dimensionality for very many matrices).

    Uses evaluate_eigenspectrum_against_null() for all dimensionality decisions,
    guaranteeing that cross-run null comparisons use the identical decision rule.
    Data is mean-centered per neuron (row) before PCA — no variance normalisation applied.

    Returns:
        If return_spectrum is False: tuple (estimated_dimensionality, shuffles_used)
        If return_spectrum is True:  tuple (estimated_dimensionality, shuffles_used,
            real_eigenvalues, null_mean, null_std) — all shape (num_components,)
    """
    if rng is None:
        rng = np.random.default_rng()

    num_neurons, num_frames = data.shape
    num_components = min(num_neurons, num_frames)

    centered_data = center(data)

    def _eigenvalues(mat: np.ndarray) -> np.ndarray:
        """Return eigenvalues of (mat @ mat.T) in descending order via eigvalsh."""
        n, t = mat.shape
        if n <= t:
            cov = mat @ mat.T / (t - 1)
            return np.sort(np.linalg.eigvalsh(cov))[::-1]
        else:
            # Gram matrix trick: identical non-zero eigenvalues, pad zeros for the rest
            gram = mat.T @ mat / (t - 1)
            eigs_small = np.sort(np.linalg.eigvalsh(gram))[::-1]
            out = np.zeros(num_components)
            out[:len(eigs_small)] = eigs_small
            return out

    real_eigenvalues = _eigenvalues(centered_data)

    null_eigenvalues_history = np.zeros((max_shuffles, num_components), dtype=float)

    patience = min_shuffles
    prev_dim = -1
    stable_count = 0

    for current_shuffle in range(max_shuffles):
        # Shuffle each neuron's time series independently (destroys correlations)
        random_indices = rng.random((num_neurons, num_frames)).argsort(axis=1)
        shuffled_data = np.take_along_axis(centered_data, random_indices, axis=1)
        null_eigenvalues_history[current_shuffle] = _eigenvalues(shuffled_data)

        number_of_completed_shuffles = current_shuffle + 1

        if number_of_completed_shuffles < min_shuffles:
            continue

        current_null_runs = null_eigenvalues_history[:number_of_completed_shuffles]
        null_mean = current_null_runs.mean(axis=0)
        null_std  = np.maximum(current_null_runs.std(axis=0, ddof=1), 1e-15)

        current_dim = evaluate_eigenspectrum_against_null(
            real_eigenvalues, null_mean, null_std, confidence_level
        )

        # Full convergence check
        z_crit = stats.norm.ppf(1.0 - (1.0 - confidence_level) / 2.0)
        n_std_clipped = np.maximum(null_std, 1e-15)
        z = (real_eigenvalues - null_mean) / n_std_clipped
        is_tail = (real_eigenvalues < 1e-10) & (null_mean < 1e-10)
        is_uncertain = (np.abs(z) <= z_crit) & ~is_tail
        if not np.any(is_uncertain):
            if return_spectrum:
                return current_dim, number_of_completed_shuffles, real_eigenvalues, null_mean, null_std
            return current_dim, number_of_completed_shuffles

        # Patience-based stopping
        if current_dim == prev_dim:
            stable_count += 1
        else:
            stable_count = 0
            prev_dim = current_dim
        if stable_count >= patience:
            if return_spectrum:
                return current_dim, number_of_completed_shuffles, real_eigenvalues, null_mean, null_std
            return current_dim, number_of_completed_shuffles

    null_mean = null_eigenvalues_history.mean(axis=0)
    null_std  = np.maximum(null_eigenvalues_history.std(axis=0, ddof=1), 1e-15)
    dim = evaluate_eigenspectrum_against_null(real_eigenvalues, null_mean, null_std, confidence_level)
    if return_spectrum:
        return dim, max_shuffles, real_eigenvalues, null_mean, null_std
    return dim, max_shuffles


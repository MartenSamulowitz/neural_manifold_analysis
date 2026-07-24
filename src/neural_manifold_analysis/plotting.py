"""Reusable plotting helpers.

All plot functions accept a matplotlib Axes object (ax) so they can be composed
into multi-panel figures. If ax is None, a new figure+axes is created.

Color conventions follow matplotlib defaults unless overridden by the caller.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import linregress

__all__ = [
    'format_xaxis_minutes',
    'plot_raster',
    'plot_correlation_matrix',
    'plot_scatter_with_regression',
    'plot_pairwise_matrix',
    'plot_cumulative_variance',
]


# ==============================================================================
# Color scaling helper (shared by the heatmap functions)
# ==============================================================================

def _resolve_color_scaling(data, vmin=None, vmax=None, cmap=None, percentile=None):
    """Decide the colour limits and colormap for a heatmap.

    Rules:
      - Explicit `vmin`/`vmax`/`cmap` always win; anything left None is derived.
      - Limits default to the data min/max, or to a percentile range if
        `percentile` is given. `percentile` may be a scalar p (interpreted as
        the (p, 100 - p) range) or a (low, high) pair of percentiles.
      - The colormap is chosen automatically when `cmap` is None: a diverging map
        ('RdBu_r') if the data straddles zero (has both signs), else 'viridis'.
      - For diverging auto-scaling with both limits left None, the limits are made
        symmetric about zero so that 0 maps to the neutral centre colour.

    Args:
        data: Array of values to be colour-mapped (NaN/inf ignored).
        vmin, vmax: Explicit limits, or None to derive them.
        cmap: Explicit colormap name, or None to pick automatically.
        percentile: Scalar or (low, high) percentiles used to derive limits.

    Returns:
        (vmin, vmax, cmap) ready to pass to imshow.
    """
    valid = np.asarray(data)[np.isfinite(data)]
    if valid.size == 0:
        return (0.0 if vmin is None else vmin,
                1.0 if vmax is None else vmax,
                cmap or 'viridis')

    if percentile is not None:
        if isinstance(percentile, (int, float)):
            low_p, high_p = float(percentile), 100.0 - float(percentile)
        else:
            low_p, high_p = percentile
        data_lo = np.percentile(valid, low_p)
        data_hi = np.percentile(valid, high_p)
    else:
        data_lo = valid.min()
        data_hi = valid.max()

    is_centered = data_lo < 0 < data_hi

    if cmap is None:
        cmap = 'RdBu_r' if is_centered else 'viridis'

    lo = data_lo if vmin is None else vmin
    hi = data_hi if vmax is None else vmax

    # Symmetric limits for diverging data when both limits are auto-derived.
    if is_centered and vmin is None and vmax is None:
        magnitude = max(abs(lo), abs(hi))
        lo, hi = -magnitude, magnitude

    return float(lo), float(hi), cmap


# ==============================================================================
# Axis Formatting
# ==============================================================================

def format_xaxis_minutes(ax, framerate: float) -> None:
    """Convert x-axis from frame indices to elapsed minutes.

    Applies a FuncFormatter that divides frame numbers by (framerate * 60)
    to display time in minutes. Useful for raster plots and time series.

    Args:
        ax: Matplotlib Axes object to format.
        framerate: Recording frame rate in Hz (frames per second).
    """
    def _frame_to_minutes(x, pos):
        return f'{x / (framerate * 60):.1f}'

    ax.xaxis.set_major_formatter(FuncFormatter(_frame_to_minutes))
    ax.set_xlabel('Time (min)')


# ==============================================================================
# Raster Plot
# ==============================================================================

def plot_raster(data: np.ndarray, ax=None, vmax: float = 0.1,
                cmap: str = 'hot', title: str = '', framerate: float | None = None,
                boundaries: np.ndarray | None = None, aspect: str = 'auto',
                xlabel: str = 'Column', ylabel: str = 'Row') -> plt.Axes:
    """Plot a 2-D array as a raster (heatmap) with optional vertical boundary lines.

    Args:
        data: Array of shape (n_rows, n_columns).
        ax: Matplotlib Axes. If None, creates a new figure.
        vmax: Colormap maximum. Values above this are clipped.
        cmap: Matplotlib colormap name.
        title: Plot title string.
        framerate: If provided, x-axis is formatted in minutes (columns / framerate).
        boundaries: 1D array of column indices, each drawn as a vertical dashed line.
        aspect: Axes aspect ratio. 'auto' stretches to fill.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        The Axes object (for further customization).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    ax.imshow(data, aspect=aspect, vmax=vmax, cmap=cmap, interpolation='none')

    if boundaries is not None:
        for boundary in boundaries:
            ax.axvline(boundary, color='cyan', linewidth=0.5, alpha=0.7, linestyle='--')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if framerate is not None:
        format_xaxis_minutes(ax, framerate)

    return ax


# ==============================================================================
# Correlation Matrix
# ==============================================================================

def plot_correlation_matrix(data: np.ndarray, ax=None, title: str = '',
                            cmap: str | None = None, vmin: float | None = None,
                            vmax: float | None = None, percentile=None,
                            colorbar: bool = True,
                            xlabel: str = '',
                            ylabel: str = '') -> plt.Axes:
    """Plot a square correlation (or distance) matrix as a heatmap with colorbar.

    Colour limits and colormap auto-adapt to the data (see below); pass explicit
    values to override.

    Args:
        data: Square array of shape (n, n).
        ax: Matplotlib Axes. If None, creates a new figure.
        title: Plot title.
        cmap: Colormap. If None, chosen automatically — 'RdBu_r' when the data
            straddles zero (e.g. signed correlations), else 'viridis'.
        vmin, vmax: Colormap limits. If None, derived from the data (min/max, or
            `percentile`); for auto-diverging data they are made symmetric about 0.
        percentile: Scalar p (→ (p, 100-p)) or a (low, high) pair of percentiles
            used to derive the limits robustly instead of the raw min/max.
        colorbar: Whether to add a colorbar.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        The Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))

    vmin, vmax, cmap = _resolve_color_scaling(data, vmin, vmax, cmap, percentile)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        plt.colorbar(im, cax=cax)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


# ==============================================================================
# Scatter with Regression
# ==============================================================================

def plot_scatter_with_regression(x: np.ndarray, y: np.ndarray, ax=None,
                                 xlabel: str = '', ylabel: str = '',
                                 title: str = '', alpha: float = 0.7,
                                 color: str | None = None,
                                 show_regression: bool = True) -> plt.Axes:
    """Scatter plot with optional linear regression line and R² annotation.

    If regression is enabled and there are enough unique x-values, a best-fit
    line is overlaid with an annotation showing R² and p-value.

    Args:
        x: 1D array — independent variable.
        y: 1D array — dependent variable (same length as x).
        ax: Matplotlib Axes. If None, creates a new figure.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Plot title.
        alpha: Scatter point transparency.
        color: Scatter point color. None uses matplotlib default.
        show_regression: Whether to overlay a linear regression line.

    Returns:
        The Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(x, y, alpha=alpha, color=color, s=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if show_regression and len(x) > 1 and np.unique(x).size > 1:
        slope, intercept, r_value, p_value, _ = linregress(x, y)
        x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color='red', linewidth=1.5,
                label=f'$R^2$={r_value**2:.3f}, p={p_value:.2g}')
        ax.legend(fontsize=9)

    return ax


# ==============================================================================
# Pairwise Matrix (generic heatmap, optional cell annotations and stats)
# ==============================================================================

def plot_pairwise_matrix(matrix: np.ndarray, ax=None, title: str = '',
                         cmap: str | None = None, colorbar_label: str = '',
                         xlabel: str = '', ylabel: str = '',
                         vmin: float | None = None, vmax: float | None = None,
                         percentile=None,
                         aspect: str = 'equal', tick_labels=None,
                         annotate: bool = False, count_matrix: np.ndarray | None = None,
                         print_stats: bool = False) -> plt.Axes:
    """Plot a pairwise metric matrix as a heatmap.

    Suitable for distance matrices, subspace-rotation matrices, similarity
    matrices, etc. Colour limits and colormap auto-adapt to the data (see below);
    optionally annotates each cell with its value and prints summary statistics.

    Args:
        matrix: 2-D array; typically square.
        ax: Matplotlib Axes. If None, creates a new figure.
        title: Plot title.
        cmap: Colormap. If None, chosen automatically — 'RdBu_r' when the data
            straddles zero (centered/signed), else 'viridis'.
        colorbar_label: Label for the colorbar.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        vmin, vmax: Colorbar limits. If None, derived from the data (min/max, or
            `percentile`); for auto-diverging data they are made symmetric about 0.
        percentile: Scalar p (→ (p, 100-p)) or a (low, high) pair of percentiles
            used to derive the limits robustly instead of the raw min/max.
        aspect: Axes aspect ratio ('equal' or 'auto').
        tick_labels: Labels for both x and y ticks. If None, uses integer indices.
        annotate: If True, draw each non-NaN value in its cell (text color flips
            around the midpoint for contrast). `count_matrix`, if given, adds an
            "n=<count>" line under each value.
        count_matrix: Optional same-shape array of counts shown when `annotate`.
        print_stats: If True, print summary stats of the values. For a square
            matrix these are the upper off-diagonal entries; otherwise all valid
            entries.

    Returns:
        The Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    lo, hi, cmap = _resolve_color_scaling(matrix, vmin, vmax, cmap, percentile)
    im = ax.imshow(matrix, cmap=cmap, aspect=aspect, vmin=lo, vmax=hi)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    plt.colorbar(im, cax=cax, label=colorbar_label)

    n_rows, n_cols = matrix.shape
    if tick_labels is not None:
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(tick_labels, fontsize=9)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    valid_vals = matrix[~np.isnan(matrix)]

    if annotate and len(valid_vals) > 0:
        mid_val = (lo + hi) / 2
        for i in range(n_rows):
            for j in range(n_cols):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                txt = f'{val:.2f}'
                if count_matrix is not None:
                    txt += f'\nn={int(count_matrix[i, j])}'
                ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                        color='white' if val < mid_val else 'black')

    if print_stats and len(valid_vals) > 0:
        if n_rows == n_cols:
            idx_i, idx_j = np.triu_indices(n_rows, k=1)
            stat_vals = matrix[idx_i, idx_j]
            stat_vals = stat_vals[~np.isnan(stat_vals)]
            suffix = " (upper off-diagonal)"
        else:
            stat_vals = valid_vals
            suffix = " (all valid elements)"

        if len(stat_vals) > 0:
            print(f"Stats for {title or 'Matrix'}{suffix}:")
            print(f"  Mean:   {np.mean(stat_vals):.4f}")
            print(f"  Median: {np.median(stat_vals):.4f}")
            print(f"  Std:    {np.std(stat_vals, ddof=1):.4f}")
            print(f"  Min:    {np.min(stat_vals):.4f}")
            print(f"  Max:    {np.max(stat_vals):.4f}\n")

    return ax


# ==============================================================================
# Cumulative Variance
# ==============================================================================

def plot_cumulative_variance(pca_object, ax=None, threshold: float = 0.7,
                              title: str = '',
                              max_components: int | None = None) -> plt.Axes:
    """Plot cumulative explained variance from a fitted PCA object.

    Shows a line plot of cumulative variance ratio with a horizontal threshold
    line and a vertical line at the dimensionality (number of PCs needed to
    reach the threshold).

    Args:
        pca_object: Fitted sklearn PCA object with .explained_variance_ratio_.
        ax: Matplotlib Axes. If None, creates a new figure.
        threshold: Variance threshold (e.g., 0.7 for 70%).
        title: Plot title.
        max_components: If set, truncate the x-axis to this many components.

    Returns:
        The Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    cumulative_var = np.cumsum(pca_object.explained_variance_ratio_)
    n_components = np.arange(1, len(cumulative_var) + 1)

    ax.plot(n_components, cumulative_var, linewidth=2)

    # Threshold line
    ax.axhline(threshold, linestyle='--', color='grey', alpha=0.7,
               label=f'Threshold = {threshold}')

    # Find dimensionality
    above_threshold = np.where(cumulative_var >= threshold)[0]
    if len(above_threshold) > 0:
        dimensionality = above_threshold[0] + 1  # 1-indexed
        ax.axvline(dimensionality, linestyle='--', color='grey', alpha=0.5)
        ax.set_title(title or f'Dimensionality = {dimensionality}')

    if max_components:
        ax.set_xlim(0, max_components)
    else:
        ax.set_xlim(0, len(cumulative_var))

    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.legend(fontsize=9)

    return ax

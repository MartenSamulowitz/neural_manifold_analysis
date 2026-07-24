# neural_manifold_analysis

Reusable primitives for **neural population geometry & manifold analysis** —
normalization, PCA, dimensionality estimation (parallel analysis, participation
ratio, thresholding), pairwise metrics, Procrustes alignment, sampling &
re-binning, IO, and plotting helpers.

This is the **shared toolbox** reused across individual paper projects. It is
deliberately domain-agnostic: it knows about matrices, geometry, and spikes in
the abstract — not about any particular experiment, belt length, or data layout.
Paper-specific preprocessing and configuration live in each paper's own repo.

## Installation

The package declares its own dependencies (numpy, scipy, scikit-learn,
matplotlib), so installing it pulls those in automatically — you do **not** need
a separate requirements list.

### Step 1 — Get the code

```bash
git clone https://github.com/MartenSamulowitz/neural_manifold_analysis.git
cd neural_manifold_analysis
```

### Step 2 — Create an environment

Use a fresh, isolated environment so this package's dependencies don't clash
with your other work. Either tool works — pick one:

```bash
# Option A — conda / mamba
conda create -n nma python=3.11
conda activate nma

# Option B — plain venv
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### Step 3 — Install the package

```bash
# Editable install (recommended while developing — edits take effect immediately):
pip install -e .

# ...or a regular install if you just want to use it:
pip install .
```

Installing directly from GitHub without cloning also works:

```bash
pip install git+https://github.com/MartenSamulowitz/neural_manifold_analysis.git
```

### Step 4 — Use it

```python
import neural_manifold_analysis as nma

order = nma.pca_angle_sorting(matrix)      # sort features by PC1–PC2 loading angle
dim   = nma.participation_ratio(matrix)    # effective dimensionality of the data
```

For live editing inside notebooks (edits to the source apply without restarting
the kernel):

```python
%load_ext autoreload
%autoreload 2
import neural_manifold_analysis as nma
```

## Modules

| Module | Contents |
|--------|----------|
| `normalizations` | Centering, standardization, L1/L2 & whole-array norms, symmetric-matrix normalization |
| `pca` | Fit PCA, sort features by loading angle, extract top-k bases |
| `dimensionality` | Parallel analysis, participation ratio, thresholding, subsampled estimation |
| `pairwise_metrics` | Atomic two-item metrics (rank-order, Jaccard, Procrustes alignment) |
| `aggregation_utilities` | Build N×N matrices from a metric; regress one matrix on another |
| `sampling_utilities` | Subsample, shuffle, CDF re-binning, upper-triangle extraction |
| `plotting` | Rasters, heatmaps, scatter + regression, cumulative-variance |
| `io` | Generic pickle artifact store (`ArtifactStore`) + dict inspection |

Each module imports exactly what it uses and declares its public API via
`__all__`; `import neural_manifold_analysis as nma` re-exports all of them.

## Status

Extracted from the internal `hgeom` package (v0.1.0). API not yet stable.
Project-specific configuration and data loaders deliberately live in each paper
repo, not here.

## License

MIT — see [LICENSE](LICENSE).

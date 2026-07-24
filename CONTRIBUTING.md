# Contributing / extending `neural_manifold_analysis`

This package is a layered toolbox of small, **generic** primitives. Keep it that
way: every function should be domain-agnostic — it knows about arrays, matrices,
and geometry, never about a specific experiment or file layout. Project-specific code belongs in the project that *uses*
the package, not here.

## The layering (don't create import cycles)

Modules sit in layers; internal imports only ever point **downward**:

```
plotting     aggregation_utilities          ← compose lower layers / visualize
                    │
dimensionality ──→ pca ──→ normalizations    ← build on the atomics
       └─────────────────→ sampling_utilities
io                                           ← standalone infrastructure
normalizations   pairwise_metrics           ← atomics: no internal dependencies
sampling_utilities
```

**Rule:** an *atomic* module (`normalizations`, `pca`, `sampling_utilities`,
`pairwise_metrics`, `io`) must not import from a *composed* one
(`dimensionality`, `aggregation_utilities`, `plotting`). If you find yourself
wanting that, the function is probably in the wrong module.

## Adding a function to an existing module

1. Write it as an **atomic** function where possible: one clear job, pure
   input → output, no file I/O, no plotting. Data arrays are
   `(n_features, n_samples)` unless there's a good reason otherwise.
2. Give it a docstring: one-line summary, then `Args:` / `Returns:`, matching the
   style of its neighbours.
3. **Add its name to that module's `__all__`.** `__all__` is the single source of
   truth for the public API — if a name isn't in it, `import nma` won't expose it
   (and internal helpers stay private, which is what you want).
4. That's all: `__init__.py` re-exports each module with `from .module import *`,
   so anything in `__all__` is automatically reachable as `nma.your_function`.

## Adding a whole new module (`.py` file)

1. Create `src/neural_manifold_analysis/<name>.py` with a module docstring and an
   `__all__` list.
2. Import only from **lower** layers (see the diagram), using **relative**
   imports: `from .normalizations import center`.
3. Add one line to `src/neural_manifold_analysis/__init__.py`:
   `from .<name> import *` — placed after the modules it depends on.
4. Add a row to the **Modules** table in `README.md`.
5. Add the module to `tests/test_smoke.py` so it's covered by the import check.

## The pairwise-metric convention

Metrics in `pairwise_metrics` take exactly two items: `(a, b) -> scalar`. If a
metric needs extra parameters, **don't** change the aggregator — bind them with
`functools.partial` at the call site so the metric still presents a two-argument
interface:

```python
from functools import partial
from neural_manifold_analysis.pairwise_metrics import rank_order_correlation

metric = partial(rank_order_correlation, circular=True)
M = nma.compute_pairwise_matrix(items, metric)
```

## Before you commit

```bash
pip install -e ".[dev]"        # installs the package + pytest
pytest                          # smoke test: everything imports
python -m pyflakes src          # optional: no undefined names / unused imports
```

Keep the public surface honest: no unused imports, no names in `__all__` that the
module doesn't define, and update `README.md` whenever the module list changes.

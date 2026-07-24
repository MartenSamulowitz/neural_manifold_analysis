"""Smoke test: the package and every submodule import cleanly."""


def test_top_level_import():
    import neural_manifold_analysis  # noqa: F401


def test_submodules_import():
    from neural_manifold_analysis import (  # noqa: F401
        aggregation_utilities,
        dimensionality,
        io,
        normalizations,
        pairwise_metrics,
        pca,
        plotting,
        sampling_utilities,
    )

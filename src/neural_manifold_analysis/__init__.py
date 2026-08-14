"""Neural population geometry & manifold-analysis primitives.

Import the whole public API in one line:

    import neural_manifold_analysis as nma

    order = nma.pca_angle_sorting(matrix)
    dim   = nma.participation_ratio(pca)

For live editing inside notebooks:

    %load_ext autoreload
    %autoreload 2
    import neural_manifold_analysis as nma

Each submodule owns its public API via its own ``__all__``; this file simply
re-exports them so everything is reachable from the top-level ``nma`` namespace.
"""

from .io import *
from .normalizations import *
from .pairwise_metrics import *
from .sampling_utilities import *
from .pca import *
from .dimensionality import *
from .aggregation_utilities import *
from .plotting import *

__version__ = "0.1.0"

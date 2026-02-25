"""geolatent — Geometry-aware, model-intelligent 3-D ML visualisations.

geolatent is a production-grade Python library that provides semantic 3-D
visualisation primitives specifically designed for machine learning workflows.
It is *not* a thin wrapper around Plotly; instead it operates as an abstraction
layer that understands model decision geometry, high-dimensional feature spaces,
and embedding manifold structure.

Key capabilities
----------------
* :func:`visualize_decision_geometry` — project a classifier's decision function
  to 3-D via PCA inverse-transform and render probability isosurfaces, class
  scatter clouds, and Mahalanobis confidence ellipsoids in a single call.

* :func:`inspect_latent_space` — visualise the geometric structure of arbitrary
  high-dimensional embeddings (transformer hidden states, VAE latent codes,
  word vectors, etc.) with configurable projection (PCA / t-SNE / UMAP),
  confidence ellipsoids, and convex-hull cluster boundaries.

Both functions produce interactive Plotly figures styled with a dark-scientific
theme and are ready for inline display in Google Colab and Jupyter notebooks.

Quick start
-----------
::

    from sklearn.svm import SVC
    from sklearn.datasets import make_classification
    from geolatent import visualize_decision_geometry

    X, y = make_classification(n_samples=300, n_features=15, n_classes=3,
                               n_informative=8, random_state=0)
    model = SVC(kernel="rbf", probability=True).fit(X, y)

    fig = visualize_decision_geometry(model, X, y)
    fig.show()

See Also
--------
* :class:`VisualizationConfig` — master configuration dataclass.
* :data:`DARK_SCIENTIFIC` — the default dark theme instance.
* ``examples/colab_demo.ipynb`` — end-to-end usage notebook.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__: str = version("geolatent")
except PackageNotFoundError:
    __version__ = "0.1.0"

__author__ = "GeoLatent Contributors"
__license__ = "MIT"

# Public API
from .api.decision import visualize_decision_geometry
from .api.latent import inspect_latent_space

# Configuration
from .config.themes import (
    ColorPalette,
    DARK_SCIENTIFIC,
    ProjectionConfig,
    RenderConfig,
    VisualizationConfig,
)

# Core building blocks (for advanced / custom pipelines)
from .core.geometry import GeometryUtils
from .core.mesh_builder import MeshBuilder, PredictionMesh
from .core.projector import DimensionalityProjector, ProjectionResult

# Rendering primitives
from .rendering.overlays import DataOverlay
from .rendering.scene import Scene3D
from .rendering.surfaces import DecisionSurfaceRenderer

__all__ = [
    # Primary API
    "visualize_decision_geometry",
    "inspect_latent_space",
    # Configuration
    "VisualizationConfig",
    "ColorPalette",
    "RenderConfig",
    "ProjectionConfig",
    "DARK_SCIENTIFIC",
    # Core
    "DimensionalityProjector",
    "ProjectionResult",
    "MeshBuilder",
    "PredictionMesh",
    "GeometryUtils",
    # Rendering
    "Scene3D",
    "DecisionSurfaceRenderer",
    "DataOverlay",
]

"""Core computation sub-package for geolatent.

Contains the dimensionality-reduction pipeline, prediction-mesh builder,
and geometric utility functions.
"""

from .geometry import GeometryUtils
from .mesh_builder import MeshBuilder, PredictionMesh
from .projector import DimensionalityProjector, ProjectionResult

__all__ = [
    "DimensionalityProjector",
    "ProjectionResult",
    "MeshBuilder",
    "PredictionMesh",
    "GeometryUtils",
]

"""Rendering sub-package for geolatent.

Contains the dark-scientific scene manager, decision-surface renderer,
and data-overlay generator.
"""

from .overlays import DataOverlay
from .scene import Scene3D
from .surfaces import DecisionSurfaceRenderer

__all__ = [
    "Scene3D",
    "DecisionSurfaceRenderer",
    "DataOverlay",
]

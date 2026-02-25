"""Prediction-mesh construction for decision-surface rendering.

The central abstraction is :class:`MeshBuilder`, which:

1. Creates a regular 3-D grid spanning the convex region of the projected data
   with a configurable padding margin.
2. Maps each grid vertex back to the original feature space via the PCA inverse
   transform stored in a fitted :class:`DimensionalityProjector`.
3. Queries the model for predicted class labels and, when available, class
   probabilities at every grid vertex.
4. Returns a :class:`PredictionMesh` bundle consumed by the rendering layer.

Notes
-----
* Grid size scales as ``resolution³``.  At ``resolution=30`` this yields
  27 000 points; at ``resolution=50`` it yields 125 000.  For most scikit-learn
  estimators the inference time remains well below 2 s for resolution ≤ 40.
* Batched inference (``batch_size`` parameter) is available to control memory
  pressure for large neural networks or kernel SVMs.
* Detection of regression vs. classification is heuristic: if the model output
  is floating-point with more than 20 unique values we treat it as a regression
  target and render a continuous scalar field rather than class volumes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .projector import DimensionalityProjector


# Result container


@dataclass
class PredictionMesh:
    """Outputs of a decision-surface prediction sweep over a 3-D grid.

    Attributes
    ----------
    x : np.ndarray of shape (resolution**3,)
        Flattened x-coordinates of the grid vertices in projected space.
    y : np.ndarray of shape (resolution**3,)
        Flattened y-coordinates.
    z : np.ndarray of shape (resolution**3,)
        Flattened z-coordinates.
    predictions : np.ndarray of shape (resolution**3,)
        Predicted class index (integer) or regression target at each vertex.
    probabilities : np.ndarray of shape (resolution**3, n_classes) or None
        Per-class probability at each vertex; ``None`` when the model does not
        expose ``predict_proba``.
    grid_shape : tuple of 3 ints
        Logical shape ``(resolution, resolution, resolution)`` of the 3-D grid.
    n_classes : int
        Number of unique predicted class labels (meaningful only for classifiers).
    unique_classes : np.ndarray
        Sorted array of unique class labels found in ``predictions``.
    bounds : np.ndarray of shape (3, 2)
        Per-axis bounding box: ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]``.
    is_regression : bool
        ``True`` when the model output is treated as a continuous regression value.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    grid_shape: Tuple[int, int, int]
    n_classes: int
    unique_classes: np.ndarray
    bounds: np.ndarray
    is_regression: bool = False


# Mesh builder


class MeshBuilder:
    """Constructs prediction meshes for 3-D decision-surface rendering.

    Parameters
    ----------
    resolution : int
        Number of grid points per spatial dimension.  Total vertex count equals
        ``resolution³``.  Default 30 provides smooth surfaces for most models
        with sub-second inference time.
    padding_fraction : float
        Fractional extension applied beyond the data bounding box on each axis.
        A value of 0.12 extends the grid by 12 % of the data range on every
        side, preventing clipping at the edges of the scatter cloud.
    batch_size : int or None
        Maximum number of points passed to the model in a single ``predict``
        call.  ``None`` infers all points at once.  Set to e.g. 4096 for
        memory-constrained models.

    Examples
    --------
    >>> from geolatent.core.mesh_builder import MeshBuilder
    >>> builder = MeshBuilder(resolution=25)
    >>> mesh = builder.build_prediction_mesh(clf, projector, X_3d)
    >>> mesh.probabilities.shape   # doctest: +SKIP
    (15625, 2)
    """

    def __init__(
        self,
        resolution: int = 30,
        padding_fraction: float = 0.12,
        batch_size: Optional[int] = None,
    ) -> None:
        if resolution < 5:
            raise ValueError("resolution must be >= 5.")
        if resolution > 100:
            warnings.warn(
                f"resolution={resolution} creates {resolution ** 3:,} grid vertices, "
                "which may cause slow inference for complex models.",
                UserWarning,
                stacklevel=2,
            )
        self.resolution = resolution
        self.padding_fraction = padding_fraction
        self.batch_size = batch_size

    # Public interface

    def build_prediction_mesh(
        self,
        model: object,
        projector: DimensionalityProjector,
        X_proj: np.ndarray,
    ) -> PredictionMesh:
        """Build a prediction mesh for *model* over the region spanned by *X_proj*.

        Parameters
        ----------
        model : sklearn-compatible estimator
            Must implement at least ``predict(X)``.
        projector : DimensionalityProjector
            Fitted projector that supports ``inverse_transform`` (i.e., PCA).
        X_proj : np.ndarray of shape (n_samples, 3)
            Training data in projected 3-D space, used to define the bounding box.

        Returns
        -------
        mesh : PredictionMesh

        Raises
        ------
        ValueError
            If *projector* does not support ``inverse_transform``.
        """
        if not projector.supports_inverse_transform:
            raise ValueError(
                "decision-surface rendering requires a PCA projector; "
                "the fitted projector does not support inverse_transform."
            )

        bounds = self._compute_bounds(X_proj)
        grid_pts_proj = self._build_grid(bounds)                         # (N³, 3)
        grid_pts_orig = projector.inverse_transform(grid_pts_proj)       # (N³, n_feat)

        predictions, probabilities, is_regression = self._query_model(
            model, grid_pts_orig
        )

        unique_classes = np.unique(predictions)
        n_classes = int(len(unique_classes))
        res = self.resolution

        return PredictionMesh(
            x=grid_pts_proj[:, 0],
            y=grid_pts_proj[:, 1],
            z=grid_pts_proj[:, 2],
            predictions=predictions,
            probabilities=probabilities,
            grid_shape=(res, res, res),
            n_classes=n_classes,
            unique_classes=unique_classes,
            bounds=bounds,
            is_regression=is_regression,
        )

    # Private helpers

    def _compute_bounds(self, X_proj: np.ndarray) -> np.ndarray:
        """Return padded per-axis bounding box, shape (3, 2)."""
        mins = X_proj.min(axis=0)
        maxs = X_proj.max(axis=0)
        ranges = maxs - mins
        # protect against degenerate (flat) axes
        ranges = np.where(ranges < 1e-8, 1.0, ranges)
        pad = ranges * self.padding_fraction
        return np.column_stack([mins - pad, maxs + pad])

    def _build_grid(self, bounds: np.ndarray) -> np.ndarray:
        """Create a regular 3-D Cartesian grid within *bounds*.

        Returns
        -------
        pts : np.ndarray of shape (resolution**3, 3)
        """
        axes = [
            np.linspace(bounds[i, 0], bounds[i, 1], self.resolution)
            for i in range(3)
        ]
        # indexing='ij' keeps axis order consistent with (x, y, z)
        g0, g1, g2 = np.meshgrid(*axes, indexing="ij")
        pts = np.column_stack([g0.ravel(), g1.ravel(), g2.ravel()])
        return pts

    def _query_model(
        self,
        model: object,
        X_orig: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
        """Run model inference on *X_orig*.

        Returns
        -------
        predictions : np.ndarray of shape (N,)
        probabilities : np.ndarray of shape (N, n_classes) or None
        is_regression : bool
        """
        predictions = self._batched_predict(model, X_orig)
        is_regression = self._detect_regression(predictions)
        probabilities: Optional[np.ndarray] = None

        if not is_regression and hasattr(model, "predict_proba"):
            try:
                probabilities = self._batched_predict_proba(model, X_orig)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"predict_proba failed with: {exc!r}.  "
                    "Confidence isosurfaces will not be rendered.",
                    UserWarning,
                    stacklevel=3,
                )

        return predictions, probabilities, is_regression

    def _batched_predict(self, model: object, X: np.ndarray) -> np.ndarray:
        if self.batch_size is None or len(X) <= self.batch_size:
            return np.asarray(model.predict(X))  # type: ignore[attr-defined]
        return np.concatenate(
            [
                np.asarray(model.predict(X[i : i + self.batch_size]))  # type: ignore
                for i in range(0, len(X), self.batch_size)
            ]
        )

    def _batched_predict_proba(self, model: object, X: np.ndarray) -> np.ndarray:
        if self.batch_size is None or len(X) <= self.batch_size:
            return np.asarray(model.predict_proba(X))  # type: ignore[attr-defined]
        return np.concatenate(
            [
                np.asarray(model.predict_proba(X[i : i + self.batch_size]))  # type: ignore
                for i in range(0, len(X), self.batch_size)
            ]
        )

    @staticmethod
    def _detect_regression(predictions: np.ndarray) -> bool:
        """Heuristic: floating-point output with many unique values → regression."""
        if np.issubdtype(predictions.dtype, np.floating):
            return int(len(np.unique(predictions))) > 20
        return False

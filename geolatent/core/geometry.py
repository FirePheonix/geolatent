"""Geometric utilities for geolatent.

Provides analytical geometry operations used to enrich decision-boundary
visualisations with class-level structural overlays:

* **Confidence ellipsoids** — parametric surfaces derived from the class-
  conditional covariance matrix, scaled to a chosen Mahalanobis-distance
  contour (analogous to a 1-σ / 2-σ region for a Gaussian assumption).
* **Class centroids** — arithmetic means of class point clouds, rendered
  as star markers that anchor the viewer's spatial reference frame.
* **Convex hull traces** (optional, computationally cheap for n ≤ 1000) —
  rendered as transparent mesh surfaces that bound each class cluster.

All methods in :class:`GeometryUtils` operate exclusively in the 3-D projected
space returned by :class:`~geolatent.core.projector.DimensionalityProjector`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from scipy.linalg import cholesky, LinAlgError
from scipy.stats import chi2

from ..config.themes import VisualizationConfig


# Geometry utilities


class GeometryUtils:
    """Collection of geometric analysis and Plotly trace generators.

    All methods accept the 3-D projected coordinate array *X_proj* and the
    label vector *y*, and return lists of fully configured Plotly traces that
    can be added directly to a :class:`~geolatent.rendering.scene.Scene3D`.

    Parameters
    ----------
    config : VisualizationConfig
        Master configuration; colour palette and render settings are read here.
    """

    def __init__(self, config: VisualizationConfig) -> None:
        self.config = config

    # Confidence ellipsoids

    def compute_class_ellipsoids(
        self,
        X_proj: np.ndarray,
        y: np.ndarray,
        *,
        confidence: float = 0.90,
        n_grid: int = 40,
        class_names: Optional[Dict] = None,
    ) -> List[go.Surface]:
        """Generate parametric ellipsoid traces for each class.

        The ellipsoid is derived from the empirical covariance matrix of the
        class subset and scaled so that it encloses *confidence* percent of
        samples under a multivariate-Gaussian assumption.

        Parameters
        ----------
        X_proj : np.ndarray of shape (n_samples, 3)
            Projected data coordinates.
        y : np.ndarray of shape (n_samples,)
            Class label vector.
        confidence : float
            Fraction of the distribution to enclose (e.g., 0.90 → 90 % region).
        n_grid : int
            Resolution of the parametric ellipsoid surface mesh.
        class_names : dict, optional
            Mapping from class label to display string.

        Returns
        -------
        traces : list of go.Surface
        """
        colors = self.config.colors.class_colors
        opacity = self.config.render.ellipsoid_opacity
        unique_classes = np.unique(y)
        traces: List[go.Surface] = []

        # Chi-squared quantile for the desired confidence level (3 DOF)
        scale = float(np.sqrt(chi2.ppf(confidence, df=3)))

        for idx, cls in enumerate(unique_classes):
            mask = y == cls
            pts = X_proj[mask]
            if len(pts) < 4:
                continue  # too few points to estimate a covariance

            mean = pts.mean(axis=0)
            cov = np.cov(pts, rowvar=False)

            try:
                L = cholesky(cov * (scale ** 2), lower=True)
            except LinAlgError:
                # Fall back to regularised covariance
                cov_reg = cov + np.eye(3) * 1e-6 * np.trace(cov)
                try:
                    L = cholesky(cov_reg * (scale ** 2), lower=True)
                except LinAlgError:
                    continue  # skip degenerate class

            # Parametric unit sphere → ellipsoid
            u = np.linspace(0, 2 * np.pi, n_grid)
            v = np.linspace(0, np.pi, n_grid)
            sphere_x = np.outer(np.cos(u), np.sin(v))
            sphere_y = np.outer(np.sin(u), np.sin(v))
            sphere_z = np.outer(np.ones(n_grid), np.cos(v))

            sphere_pts = np.stack(
                [sphere_x.ravel(), sphere_y.ravel(), sphere_z.ravel()], axis=1
            )  # (n_grid², 3)

            ellipsoid_pts = (L @ sphere_pts.T).T + mean  # (n_grid², 3)

            ex = ellipsoid_pts[:, 0].reshape(n_grid, n_grid)
            ey = ellipsoid_pts[:, 1].reshape(n_grid, n_grid)
            ez = ellipsoid_pts[:, 2].reshape(n_grid, n_grid)

            color = colors[idx % len(colors)]
            label = self._class_label(cls, class_names)

            traces.append(
                go.Surface(
                    x=ex,
                    y=ey,
                    z=ez,
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    opacity=opacity,
                    name=f"{label} — {int(confidence * 100)}% ellipsoid",
                    hoverinfo="name",
                    lighting=dict(
                        ambient=0.7,
                        diffuse=0.4,
                        specular=0.1,
                        roughness=0.9,
                    ),
                    showlegend=True,
                )
            )

        return traces

    # Class centroids

    def compute_class_centroids(
        self,
        X_proj: np.ndarray,
        y: np.ndarray,
        *,
        class_names: Optional[Dict] = None,
    ) -> go.Scatter3d:
        """Build a single Scatter3d trace for all class centroids.

        Parameters
        ----------
        X_proj : np.ndarray of shape (n_samples, 3)
        y : np.ndarray of shape (n_samples,)
        class_names : dict, optional

        Returns
        -------
        trace : go.Scatter3d
            Star-shaped markers at the class-mean coordinates.
        """
        colors = self.config.colors.class_colors
        unique_classes = np.unique(y)

        cx, cy, cz, c_colors, c_labels = [], [], [], [], []

        for idx, cls in enumerate(unique_classes):
            mask = y == cls
            mean = X_proj[mask].mean(axis=0)
            cx.append(float(mean[0]))
            cy.append(float(mean[1]))
            cz.append(float(mean[2]))
            c_colors.append(colors[idx % len(colors)])
            c_labels.append(self._class_label(cls, class_names))

        return go.Scatter3d(
            x=cx,
            y=cy,
            z=cz,
            mode="markers+text",
            marker=dict(
                symbol="diamond",
                size=self.config.render.centroid_marker_size,
                color=c_colors,
                line=dict(
                    color=self.config.colors.centroid_color,
                    width=2,
                ),
            ),
            text=c_labels,
            textposition="top center",
            textfont=dict(
                color=self.config.colors.text,
                size=11,
                family=self.config.render.font_family,
            ),
            name="Class centroids",
            hovertemplate=(
                "<b>%{text}</b><br>"
                "x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>"
            ),
        )

    # Convex hulls

    def compute_convex_hull_traces(
        self,
        X_proj: np.ndarray,
        y: np.ndarray,
        *,
        class_names: Optional[Dict] = None,
    ) -> List[go.Mesh3d]:
        """Generate convex-hull surface traces for each class cluster.

        Parameters
        ----------
        X_proj : np.ndarray of shape (n_samples, 3)
        y : np.ndarray of shape (n_samples,)
        class_names : dict, optional

        Returns
        -------
        traces : list of go.Mesh3d
            One mesh per class, rendered at very low opacity.
        """
        try:
            from scipy.spatial import ConvexHull
        except ImportError:
            return []

        colors = self.config.colors.class_colors
        unique_classes = np.unique(y)
        traces: List[go.Mesh3d] = []

        for idx, cls in enumerate(unique_classes):
            pts = X_proj[y == cls]
            if len(pts) < 4:
                continue
            try:
                hull = ConvexHull(pts)
            except Exception:  # noqa: BLE001
                continue

            vertices = pts[hull.vertices]
            simplices = hull.simplices

            # Re-index simplices to refer to the hull.vertices subset
            vertex_map = {orig: new for new, orig in enumerate(hull.vertices)}
            ti = np.array(
                [[vertex_map[s] for s in simplex] for simplex in simplices]
            )

            color = colors[idx % len(colors)]
            label = self._class_label(cls, class_names)

            traces.append(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=ti[:, 0],
                    j=ti[:, 1],
                    k=ti[:, 2],
                    color=color,
                    opacity=0.07,
                    name=f"{label} — convex hull",
                    hoverinfo="name",
                    flatshading=True,
                    showlegend=True,
                )
            )

        return traces

    # Private helpers

    @staticmethod
    def _class_label(cls: object, class_names: Optional[Dict]) -> str:
        if class_names and cls in class_names:
            return str(class_names[cls])
        return f"Class {cls}"

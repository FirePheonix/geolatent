"""Data-point overlay rendering for geolatent.

:class:`DataOverlay` is responsible for all trace types that are layered *on
top* of decision surfaces: per-class scatter clouds, class-centroid markers,
and optional confidence-ellipsoid surfaces.

Design rationale: each rendering method returns a list of independent Plotly
traces rather than mutating an existing figure.  This keeps the overlay layer
stateless and trivially composable with scene objects produced by other modules.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

from ..config.themes import VisualizationConfig
from ..core.geometry import GeometryUtils


class DataOverlay:
    """Generates data-point and structural overlays for 3-D scenes.

    Parameters
    ----------
    config : VisualizationConfig
        Master configuration; colour palette, marker sizes, and opacity values
        are all sourced from here.

    Examples
    --------
    >>> overlay = DataOverlay(DARK_SCIENTIFIC)
    >>> traces = overlay.render_scatter(X_3d, y)
    >>> centroid = overlay.render_centroids(X_3d, y)
    """

    def __init__(self, config: VisualizationConfig) -> None:
        self.config = config
        self._geometry = GeometryUtils(config)

    # Scatter cloud

    def render_scatter(
        self,
        X_proj: np.ndarray,
        y: np.ndarray,
        *,
        class_names: Optional[Dict] = None,
        point_size_override: Optional[int] = None,
        opacity_override: Optional[float] = None,
    ) -> List[go.Scatter3d]:
        """Render a per-class scatter trace for every unique class label.

        One ``go.Scatter3d`` trace is produced per class, allowing Plotly's
        interactive legend to toggle individual classes on and off.

        Parameters
        ----------
        X_proj : np.ndarray of shape (n_samples, 3)
            3-D projected coordinates.
        y : np.ndarray of shape (n_samples,)
            Class label vector.
        class_names : dict, optional
            Mapping ``{label: display_string}``.
        point_size_override : int, optional
            Override ``config.render.marker_size``.
        opacity_override : float, optional
            Override ``config.render.scatter_opacity``.

        Returns
        -------
        traces : list of go.Scatter3d
            One trace per unique class, in sorted label order.
        """
        colors = self.config.colors.class_colors
        size = point_size_override or self.config.render.marker_size
        opacity = opacity_override or self.config.render.scatter_opacity
        unique_classes = np.unique(y)
        traces: List[go.Scatter3d] = []

        for idx, cls in enumerate(unique_classes):
            mask = y == cls
            pts = X_proj[mask]
            color = colors[idx % len(colors)]
            label = self._class_label(cls, class_names)

            # Build hover text
            hover_text = [
                f"<b>{label}</b><br>"
                f"x: {pt[0]:.4f}<br>y: {pt[1]:.4f}<br>z: {pt[2]:.4f}"
                for pt in pts
            ]

            traces.append(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="markers",
                    marker=dict(
                        size=size,
                        color=color,
                        opacity=opacity,
                        line=dict(
                            color=self.config.colors.marker_line_color,
                            width=0.5,
                        ),
                        symbol="circle",
                    ),
                    name=label,
                    legendgroup=str(cls),
                    showlegend=True,
                    text=hover_text,
                    hovertemplate="%{text}<extra></extra>",
                )
            )

        return traces

    # Class centroids

    def render_centroids(
        self,
        X_proj: np.ndarray,
        y: np.ndarray,
        *,
        class_names: Optional[Dict] = None,
    ) -> go.Scatter3d:
        """Render diamond-shaped class-centroid markers.

        Delegates to :meth:`~geolatent.core.geometry.GeometryUtils.compute_class_centroids`.

        Parameters
        ----------
        X_proj : np.ndarray of shape (n_samples, 3)
        y : np.ndarray of shape (n_samples,)
        class_names : dict, optional

        Returns
        -------
        trace : go.Scatter3d
        """
        return self._geometry.compute_class_centroids(
            X_proj, y, class_names=class_names
        )

    # Confidence ellipsoids

    def render_ellipsoids(
        self,
        X_proj: np.ndarray,
        y: np.ndarray,
        *,
        confidence: float = 0.90,
        class_names: Optional[Dict] = None,
    ) -> List[go.Surface]:
        """Render parametric confidence-ellipsoid surfaces for each class.

        Delegates to :meth:`~geolatent.core.geometry.GeometryUtils.compute_class_ellipsoids`.

        Parameters
        ----------
        X_proj : np.ndarray of shape (n_samples, 3)
        y : np.ndarray of shape (n_samples,)
        confidence : float
            Confidence level for the Mahalanobis-distance ellipsoid (default 0.90).
        class_names : dict, optional

        Returns
        -------
        traces : list of go.Surface
        """
        return self._geometry.compute_class_ellipsoids(
            X_proj, y, confidence=confidence, class_names=class_names
        )

    # Convex hulls

    def render_convex_hulls(
        self,
        X_proj: np.ndarray,
        y: np.ndarray,
        *,
        class_names: Optional[Dict] = None,
    ) -> List[go.Mesh3d]:
        """Render transparent convex-hull surfaces around each class cluster.

        Delegates to
        :meth:`~geolatent.core.geometry.GeometryUtils.compute_convex_hull_traces`.

        Parameters
        ----------
        X_proj : np.ndarray of shape (n_samples, 3)
        y : np.ndarray of shape (n_samples,)
        class_names : dict, optional

        Returns
        -------
        traces : list of go.Mesh3d
        """
        return self._geometry.compute_convex_hull_traces(
            X_proj, y, class_names=class_names
        )

    # Trajectory (optimisation path / attention walk)

    def render_trajectory(
        self,
        waypoints: np.ndarray,
        *,
        name: str = "Trajectory",
        color: Optional[str] = None,
        line_width: int = 4,
        show_waypoints: bool = True,
    ) -> List[go.BaseTraceType]:
        """Render an ordered sequence of waypoints as a 3-D polyline.

        Useful for visualising gradient-descent trajectories, attention paths,
        or other sequential processes in the projected embedding space.

        Parameters
        ----------
        waypoints : np.ndarray of shape (n_steps, 3)
            Ordered sequence of 3-D positions.
        name : str
            Legend label for the trajectory.
        color : str, optional
            Line colour (hex).  Defaults to the first class colour.
        line_width : int
            Width of the line in pixels.
        show_waypoints : bool
            Whether to overlay individual step markers.

        Returns
        -------
        traces : list of Plotly traces
        """
        waypoints = np.asarray(waypoints, dtype=np.float64)
        if waypoints.ndim != 2 or waypoints.shape[1] != 3:
            raise ValueError(
                "waypoints must be a 2-D array of shape (n_steps, 3)."
            )

        if color is None:
            color = self.config.colors.class_colors[0]

        traces: List[go.BaseTraceType] = [
            go.Scatter3d(
                x=waypoints[:, 0],
                y=waypoints[:, 1],
                z=waypoints[:, 2],
                mode="lines",
                line=dict(color=color, width=line_width),
                name=name,
                showlegend=True,
                hoverinfo="skip",
            )
        ]

        if show_waypoints:
            n = len(waypoints)
            step_colors = [
                f"rgba({int(0x58)},{int(0xa6)},{int(0xff)},{i / max(n - 1, 1):.2f})"
                for i in range(n)
            ]
            traces.append(
                go.Scatter3d(
                    x=waypoints[:, 0],
                    y=waypoints[:, 1],
                    z=waypoints[:, 2],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=list(range(n)),
                        colorscale="Blues",
                        showscale=False,
                    ),
                    name=f"{name} steps",
                    showlegend=False,
                    hovertemplate=(
                        "Step %{customdata}<br>"
                        "x: %{x:.4f}<br>y: %{y:.4f}<br>z: %{z:.4f}<extra></extra>"
                    ),
                    customdata=list(range(n)),
                )
            )

        return traces

    # Private helpers

    @staticmethod
    def _class_label(cls: object, class_names: Optional[Dict]) -> str:
        if class_names and cls in class_names:
            return str(class_names[cls])
        return f"Class {cls}"

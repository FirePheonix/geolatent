"""Decision-surface rendering for geolatent.

:class:`DecisionSurfaceRenderer` converts a :class:`~geolatent.core.mesh_builder.PredictionMesh`
into a set of Plotly traces that collectively visualise the geometry of a
model's decision function in 3-D projected space.

Three rendering strategies are implemented:

``render_probability_isosurfaces``
    For classifiers that expose ``predict_proba``.  Renders one Plotly
    ``Isosurface`` per class at the probability threshold of 0.50, producing
    clean translucent shells that exactly trace the decision boundary for each
    class in a one-vs-rest sense.  Additional shells at higher thresholds
    (0.70, 0.85) encode confidence depth.

``render_class_volumes``
    For classifiers without ``predict_proba`` (e.g., ``LinearSVC``).  Renders
    a single ``Volume`` trace coloured by predicted class index, giving a
    volumetric view of the class partition.

``render_regression_field``
    For regression models.  Renders a ``Volume`` trace with a diverging
    colorscale representing the scalar prediction field.

Strategy selection is automatic: the renderer inspects the :class:`PredictionMesh`
and chooses the most informative approach.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

from ..config.themes import VisualizationConfig
from ..core.mesh_builder import PredictionMesh


# Decision-surface renderer


class DecisionSurfaceRenderer:
    """Renders decision surfaces from a :class:`PredictionMesh`.

    Parameters
    ----------
    config : VisualizationConfig
        Master configuration; colour palette and surface opacity are read here.

    Examples
    --------
    >>> renderer = DecisionSurfaceRenderer(DARK_SCIENTIFIC)
    >>> traces = renderer.render(mesh)
    >>> for t in traces:
    ...     scene.add_trace(t)
    """

    def __init__(self, config: VisualizationConfig) -> None:
        self.config = config

    # Public dispatch

    def render(
        self,
        mesh: PredictionMesh,
        *,
        class_names: Optional[Dict] = None,
        show_confidence: bool = True,
    ) -> List[go.BaseTraceType]:
        """Auto-select and execute the most appropriate rendering strategy.

        Parameters
        ----------
        mesh : PredictionMesh
            Pre-computed prediction mesh from :class:`~geolatent.core.mesh_builder.MeshBuilder`.
        class_names : dict, optional
            Mapping from class label to display string.
        show_confidence : bool
            When ``True`` and ``mesh.probabilities`` is available, render
            nested confidence isosurfaces in addition to the boundary shell.

        Returns
        -------
        traces : list of Plotly traces
        """
        if mesh.is_regression:
            return self.render_regression_field(mesh)

        if mesh.probabilities is not None and show_confidence:
            return self.render_probability_isosurfaces(
                mesh, class_names=class_names, show_confidence=show_confidence
            )

        return self.render_class_volumes(mesh, class_names=class_names)

    # Probability isosurfaces (classifiers with predict_proba)

    def render_probability_isosurfaces(
        self,
        mesh: PredictionMesh,
        *,
        class_names: Optional[Dict] = None,
        show_confidence: bool = True,
        boundary_threshold: float = 0.50,
        confidence_thresholds: Optional[List[float]] = None,
    ) -> List[go.Isosurface]:
        """Render probability isosurfaces for each class.

        For each class *c*, the decision boundary surface (``P(class=c) = 0.5``)
        is rendered as a translucent shell.  When *show_confidence* is ``True``,
        additional shells at higher probability thresholds convey confidence depth.

        Parameters
        ----------
        mesh : PredictionMesh
            Must have ``probabilities`` populated.
        class_names : dict, optional
        show_confidence : bool
        boundary_threshold : float
            Primary isosurface probability value (default 0.50).
        confidence_thresholds : list of float, optional
            Additional isosurface levels drawn inside the boundary shell.
            Defaults to ``[0.70, 0.85]`` when *show_confidence* is ``True``.

        Returns
        -------
        traces : list of go.Isosurface
        """
        assert mesh.probabilities is not None, "probabilities must be populated"

        if confidence_thresholds is None:
            confidence_thresholds = [0.70, 0.85] if show_confidence else []

        colors = self.config.colors.class_colors
        opacity = self.config.render.surface_opacity
        traces: List[go.Isosurface] = []

        # Build threshold list: boundary first, then confidence shells (inner)
        thresholds = [boundary_threshold] + confidence_thresholds
        opacities = [opacity] + [opacity * 0.55 for _ in confidence_thresholds]

        for class_idx, cls in enumerate(mesh.unique_classes):
            # Probability of this class at each grid vertex
            prob_col = int(
                np.searchsorted(
                    np.sort(np.unique(mesh.predictions)),
                    cls,
                )
            )
            # Use the class index within probabilities matrix directly
            prob_col = class_idx
            p_values = mesh.probabilities[:, prob_col]

            color = colors[class_idx % len(colors)]
            label = self._class_label(cls, class_names)

            for thresh, op in zip(thresholds, opacities):
                # Skip if the threshold is not crossed anywhere in the field
                if p_values.max() <= thresh or p_values.min() >= thresh:
                    continue

                surface_count = 1
                is_boundary = thresh == boundary_threshold

                traces.append(
                    go.Isosurface(
                        x=mesh.x,
                        y=mesh.y,
                        z=mesh.z,
                        value=p_values,
                        isomin=float(thresh) - 1e-4,
                        isomax=float(thresh) + 1e-4,
                        surface=dict(count=surface_count, fill=1.0),
                        caps=dict(x_show=False, y_show=False, z_show=False),
                        colorscale=[[0, color], [1, color]],
                        showscale=False,
                        opacity=op,
                        name=(
                            f"{label} boundary"
                            if is_boundary
                            else f"{label} p={thresh:.0%}"
                        ),
                        hoverinfo="skip",
                        flatshading=False,
                        lighting=dict(
                            ambient=0.6,
                            diffuse=0.5,
                            specular=0.15,
                            roughness=0.7,
                            fresnel=0.2,
                        ),
                        lightposition=dict(x=1000, y=1000, z=1000),
                        showlegend=is_boundary,
                        legendgroup=str(cls),
                    )
                )

        return traces

    # Class volume (classifiers without predict_proba)

    def render_class_volumes(
        self,
        mesh: PredictionMesh,
        *,
        class_names: Optional[Dict] = None,
    ) -> List[go.BaseTraceType]:
        """Render a per-class volume for models without ``predict_proba``.

        Maps predicted class indices onto a discrete colour scale and renders
        the full volumetric class partition as a transparent ``go.Volume``.
        To give class boundaries a clear visual edge, one ``go.Isosurface`` per
        class boundary is additionally rendered.

        Parameters
        ----------
        mesh : PredictionMesh
        class_names : dict, optional

        Returns
        -------
        traces : list of Plotly traces
        """
        colors = self.config.colors.class_colors
        opacity = self.config.render.surface_opacity
        traces: List[go.BaseTraceType] = []

        # Map class labels to consecutive integers [0, n_classes)
        label_to_int = {cls: i for i, cls in enumerate(mesh.unique_classes)}
        pred_int = np.vectorize(label_to_int.get)(mesh.predictions).astype(float)

        # Build a discrete colorscale
        n = len(mesh.unique_classes)
        colorscale = []
        for i, cls in enumerate(mesh.unique_classes):
            color = colors[i % len(colors)]
            lo = i / n
            hi = (i + 1) / n
            colorscale.extend([[lo, color], [hi, color]])

        # Volume trace for overall class field
        traces.append(
            go.Volume(
                x=mesh.x,
                y=mesh.y,
                z=mesh.z,
                value=pred_int,
                isomin=0.0,
                isomax=float(n - 1),
                opacity=opacity * 0.4,
                surface_count=n,
                colorscale=colorscale,
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
                name="Decision regions",
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Isosurface at each class boundary
        for i, cls in enumerate(mesh.unique_classes):
            color = colors[i % len(colors)]
            label = self._class_label(cls, class_names)
            boundary_val = float(i) + 0.5

            if i == 0:
                continue  # no lower boundary for first class

            traces.append(
                go.Isosurface(
                    x=mesh.x,
                    y=mesh.y,
                    z=mesh.z,
                    value=pred_int,
                    isomin=boundary_val - 1e-4,
                    isomax=boundary_val + 1e-4,
                    surface=dict(count=1, fill=1.0),
                    caps=dict(x_show=False, y_show=False, z_show=False),
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    opacity=opacity,
                    name=f"{label} boundary",
                    hoverinfo="skip",
                    lighting=dict(
                        ambient=0.6,
                        diffuse=0.5,
                        specular=0.1,
                        roughness=0.8,
                    ),
                    showlegend=True,
                )
            )

        return traces

    # Regression field

    def render_regression_field(self, mesh: PredictionMesh) -> List[go.Volume]:
        """Render a continuous regression prediction field as a volumetric trace.

        Parameters
        ----------
        mesh : PredictionMesh
            A mesh whose ``is_regression`` flag is ``True``.

        Returns
        -------
        traces : list of go.Volume
        """
        values = mesh.predictions.astype(float)
        vmin, vmax = float(values.min()), float(values.max())

        return [
            go.Volume(
                x=mesh.x,
                y=mesh.y,
                z=mesh.z,
                value=values,
                isomin=vmin,
                isomax=vmax,
                opacity=self.config.render.surface_opacity * 0.6,
                surface_count=12,
                colorscale=self.config.colors.surface_colorscale,
                showscale=self.config.render.show_colorbar,
                colorbar=dict(
                    title="Prediction",
                    tickfont=dict(
                        color=self.config.colors.annotation_color,
                        family=self.config.render.font_family,
                    ),
                    titlefont=dict(
                        color=self.config.colors.text,
                        family=self.config.render.font_family,
                    ),
                    bgcolor="rgba(22,27,34,0.8)",
                    bordercolor=self.config.colors.axis_line,
                    borderwidth=1,
                ),
                caps=dict(x_show=False, y_show=False, z_show=False),
                name="Regression field",
                hoverinfo="skip",
                showlegend=True,
            )
        ]

    # Private helpers

    @staticmethod
    def _class_label(cls: object, class_names: Optional[Dict]) -> str:
        if class_names and cls in class_names:
            return str(class_names[cls])
        return f"Class {cls}"

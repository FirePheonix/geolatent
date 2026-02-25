"""3-D scene manager for geolatent.

:class:`Scene3D` wraps a Plotly ``go.Figure`` and manages:

* Application of the dark-scientific theme to every layout property.
* A fluent ``add_trace`` / ``add_traces`` interface that returns ``self``
  for method chaining.
* Camera positioning, axis labelling, and annotation injection.
* The ``render()`` finalisation step that returns the completed figure.

The scene is intentionally decoupled from domain logic (decision surfaces,
scatter overlays, etc.) to keep the rendering layer independently testable
and reusable.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import plotly.graph_objects as go

from ..config.themes import VisualizationConfig


class Scene3D:
    """Stateful 3-D Plotly figure builder with dark-scientific theming.

    Parameters
    ----------
    config : VisualizationConfig
        Master configuration; theme colours, render sizes, and camera settings
        are all read from here.

    Examples
    --------
    >>> scene = Scene3D(DARK_SCIENTIFIC)
    >>> scene.add_trace(some_trace).add_trace(another_trace)
    >>> fig = scene.render()
    >>> fig.show()
    """

    def __init__(self, config: VisualizationConfig) -> None:
        self.config = config
        self._fig = go.Figure()
        self._title: Optional[str] = None
        self._axis_labels: List[str] = ["Dim 1", "Dim 2", "Dim 3"]
        self._annotations: List[dict] = []

    # Fluent trace management

    def add_trace(self, trace: go.BaseTraceType) -> "Scene3D":
        """Add a single Plotly trace.

        Parameters
        ----------
        trace : go.BaseTraceType

        Returns
        -------
        self
        """
        self._fig.add_trace(trace)
        return self

    def add_traces(self, traces: Sequence[go.BaseTraceType]) -> "Scene3D":
        """Add multiple Plotly traces at once.

        Parameters
        ----------
        traces : sequence of go.BaseTraceType

        Returns
        -------
        self
        """
        for trace in traces:
            self._fig.add_trace(trace)
        return self

    # Metadata setters

    def set_title(self, title: str) -> "Scene3D":
        """Set the figure title.

        Parameters
        ----------
        title : str

        Returns
        -------
        self
        """
        self._title = title
        return self

    def set_axis_labels(self, labels: List[str]) -> "Scene3D":
        """Override the three axis labels.

        Parameters
        ----------
        labels : list of 3 str

        Returns
        -------
        self
        """
        if len(labels) != 3:
            raise ValueError("labels must contain exactly 3 strings.")
        self._axis_labels = labels
        return self

    def add_variance_annotation(self, explained_variance_ratio: object) -> "Scene3D":
        """Append a cumulative explained-variance annotation.

        Parameters
        ----------
        explained_variance_ratio : array-like of shape (3,)
            Per-component explained-variance ratios from PCA.

        Returns
        -------
        self
        """
        import numpy as np

        ev = np.asarray(explained_variance_ratio)
        total = float(np.sum(ev))
        per = [f"{v * 100:.1f}%" for v in ev]
        text = (
            f"PCA explained variance<br>"
            f"PC1 {per[0]}  PC2 {per[1]}  PC3 {per[2]}<br>"
            f"<b>Cumulative: {total * 100:.1f}%</b>"
        )
        self._annotations.append(
            dict(
                text=text,
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.01,
                y=0.01,
                align="left",
                font=dict(
                    color=self.config.colors.annotation_color,
                    size=10,
                    family=self.config.render.font_family,
                ),
                bgcolor="rgba(13,17,23,0.6)",
                bordercolor=self.config.colors.axis_line,
                borderwidth=1,
                borderpad=4,
            )
        )
        return self

    def add_text_annotation(self, text: str, *, x: float = 0.99, y: float = 0.99) -> "Scene3D":
        """Add a free-form text annotation in paper coordinates.

        Parameters
        ----------
        text : str
            HTML-formatted annotation text.
        x, y : float
            Paper-coordinate position (0–1).

        Returns
        -------
        self
        """
        self._annotations.append(
            dict(
                text=text,
                showarrow=False,
                xref="paper",
                yref="paper",
                x=x,
                y=y,
                align="right",
                font=dict(
                    color=self.config.colors.annotation_color,
                    size=10,
                    family=self.config.render.font_family,
                ),
                bgcolor="rgba(13,17,23,0.6)",
                bordercolor=self.config.colors.axis_line,
                borderwidth=1,
                borderpad=4,
            )
        )
        return self

    # Render (finalise and return figure)

    def render(self) -> go.Figure:
        """Apply the dark-scientific layout and return the completed figure.

        Returns
        -------
        fig : go.Figure
            A fully configured Plotly figure ready for ``fig.show()`` or
            ``fig.write_html()``.
        """
        cfg = self.config
        c = cfg.colors
        r = cfg.render

        # axis sub-dict (shared structure for all three axes)
        def _axis(label: str) -> dict:
            show = r.show_axes
            return dict(
                title=dict(
                    text=label,
                    font=dict(color=c.text, size=r.font_size, family=r.font_family),
                ),
                tickfont=dict(color=c.annotation_color, size=10, family=r.font_family),
                gridcolor=c.grid,
                showgrid=show,
                zeroline=show,
                zerolinecolor=c.axis_line,
                showline=show,
                linecolor=c.axis_line,
                backgroundcolor=c.background,
                showbackground=True,
                showspikes=False,
            )

        scene_dict = dict(
            bgcolor=c.background,
            xaxis=_axis(self._axis_labels[0]),
            yaxis=_axis(self._axis_labels[1]),
            zaxis=_axis(self._axis_labels[2]),
            camera=dict(
                eye=r.camera_eye,
                up=r.camera_up,
            ),
            aspectmode="auto",
        )

        # title
        title_obj: Union[str, dict]
        if self._title:
            title_obj = dict(
                text=self._title,
                font=dict(
                    color=c.text,
                    size=r.font_size + 4,
                    family=r.font_family,
                ),
                x=0.5,
                xanchor="center",
            )
        else:
            title_obj = ""

        # legend
        legend_dict = dict(
            bgcolor="rgba(22,27,34,0.85)",
            bordercolor=c.axis_line,
            borderwidth=1,
            font=dict(color=c.text, size=10, family=r.font_family),
            itemsizing="constant",
            tracegroupgap=4,
        )

        self._fig.update_layout(
            width=r.width,
            height=r.height,
            paper_bgcolor=c.background,
            plot_bgcolor=c.background,
            font=dict(color=c.text, family=r.font_family, size=r.font_size),
            scene=scene_dict,
            title=title_obj,
            showlegend=r.show_legend,
            legend=legend_dict,
            annotations=self._annotations,
            margin=dict(l=0, r=0, b=0, t=60 if self._title else 20),
            uirevision="geolatent",  # preserve camera on re-render
        )

        return self._fig

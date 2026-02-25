"""Configuration management for geolatent visualizations.

Provides dataclass-based theme definitions, colour palettes, rendering parameters,
and projection settings.  The module ships with a pre-built ``DARK_SCIENTIFIC``
theme modelled after the aesthetic conventions of modern ML research publications
(Weights & Biases dark mode, Papers With Code, high-quality arXiv supplementary
materials).

All configuration objects are plain Python dataclasses and support deep-copy
via the ``.copy()`` helper, making it trivial to derive customised themes from
the defaults without mutation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Colour palette


@dataclass
class ColorPalette:
    """Specification of all colours used within a single visualisation theme.

    Parameters
    ----------
    background : str
        Hex colour for the outermost canvas / paper background.
    grid : str
        Hex colour for axis gridlines and background panes.
    text : str
        Hex colour for all labels, titles, and annotations.
    axis_line : str
        Hex colour for zero-axis lines and tick marks.
    class_colors : list of str
        Ordered list of hex colours assigned to distinct classes.  Colours are
        cycled when the number of classes exceeds the list length.
    surface_colorscale : str
        Plotly-compatible named colorscale used for probability volumes and
        scalar-field surfaces.
    boundary_colorscale : str
        Colorscale used to shade diverging decision-boundary surfaces.
    marker_line_color : str
        Colour of the thin outline drawn around scatter markers.
    centroid_color : str
        Colour of the star-shaped class-centroid markers.
    annotation_color : str
        Colour of variance-explained and other textual annotations.
    """

    background: str = "#0d1117"
    grid: str = "#161b22"
    text: str = "#e6edf3"
    axis_line: str = "#30363d"
    class_colors: List[str] = field(
        default_factory=lambda: [
            "#58a6ff",  # GitHub blue
            "#3fb950",  # GitHub green
            "#f78166",  # GitHub red-orange
            "#d2a8ff",  # GitHub purple
            "#ffa657",  # GitHub amber
            "#79c0ff",  # GitHub sky
            "#56d364",  # GitHub lime
            "#ff7b72",  # GitHub coral
        ]
    )
    surface_colorscale: str = "Plasma"
    boundary_colorscale: str = "RdBu"
    marker_line_color: str = "#0d1117"
    centroid_color: str = "#f0f6fc"
    annotation_color: str = "#8b949e"


# Rendering parameters


@dataclass
class RenderConfig:
    """Low-level rendering and layout parameters.

    Parameters
    ----------
    width : int
        Canvas width in pixels.
    height : int
        Canvas height in pixels.
    surface_opacity : float
        Alpha value (0–1) for decision boundary / probability isosurfaces.
    scatter_opacity : float
        Alpha value (0–1) for data-point scatter markers.
    marker_size : int
        Diameter of scatter markers in pixels.
    centroid_marker_size : int
        Diameter of class-centroid markers in pixels.
    ellipsoid_opacity : float
        Alpha value (0–1) for confidence ellipsoid surfaces.
    show_axes : bool
        Whether to display axis labels, ticks, and gridlines.
    show_legend : bool
        Whether to show the interactive legend panel.
    show_colorbar : bool
        Whether to show the colourbar for scalar-field surfaces.
    camera_eye : dict
        Plotly camera ``eye`` dict with keys ``x``, ``y``, ``z``.
    camera_up : dict
        Plotly camera ``up`` vector dict.
    font_family : str
        Font family string passed to Plotly's ``font`` specification.
    font_size : int
        Base font size in points.
    """

    width: int = 960
    height: int = 720
    surface_opacity: float = 0.28
    scatter_opacity: float = 0.88
    marker_size: int = 5
    centroid_marker_size: int = 14
    ellipsoid_opacity: float = 0.15
    show_axes: bool = True
    show_legend: bool = True
    show_colorbar: bool = False
    camera_eye: Dict[str, float] = field(
        default_factory=lambda: {"x": 1.55, "y": 1.55, "z": 1.15}
    )
    camera_up: Dict[str, float] = field(
        default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 1.0}
    )
    font_family: str = (
        "'JetBrains Mono', 'Fira Code', 'Consolas', 'Courier New', monospace"
    )
    font_size: int = 12


# Projection / dimensionality-reduction parameters


@dataclass
class ProjectionConfig:
    """Configuration for the dimensionality-reduction pipeline.

    Parameters
    ----------
    method : {"pca", "tsne", "umap", "sensitivity"}
        Projection algorithm.  ``"pca"`` and ``"sensitivity"`` both support
        inverse-transform (required for decision-surface rendering).
        ``"sensitivity"`` finds axes the model actually cares about using
        finite-difference Jacobians.  ``"tsne"`` and ``"umap"`` are for
        latent-space visualisation only.
    n_components : int
        Target dimensionality.  Should be 3 for all 3-D visualisations.
    whiten : bool
        Whether to whiten the PCA output (unit variance per component).
    tsne_perplexity : float
        Perplexity hyper-parameter for t-SNE.  Should be less than n_samples.
    tsne_learning_rate : float
        Learning rate for t-SNE gradient descent.
    tsne_n_iter : int
        Maximum number of iterations for t-SNE optimisation.
    umap_n_neighbors : int
        Number of neighbours used by UMAP to build the local topology.
    umap_min_dist : float
        Minimum distance between embedded points for UMAP.
    random_state : int
        Global random seed for reproducibility across all stochastic operations.
    scale_input : bool
        Whether to standardise features (zero-mean, unit-variance) before
        applying the chosen projection.  Strongly recommended for PCA.
    """

    method: str = "pca"
    n_components: int = 3
    whiten: bool = False
    tsne_perplexity: float = 30.0
    tsne_learning_rate: float = 200.0
    tsne_n_iter: int = 1000
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    random_state: int = 42
    scale_input: bool = True
    # Sensitivity projection
    sensitivity_n_subsample: int = 300
    sensitivity_eps: float = 1e-2


# Master configuration container


@dataclass
class VisualizationConfig:
    """Top-level configuration container for geolatent.

    Combines colour, rendering, and projection sub-configurations into a single
    immutable-style object that can be passed to any public API function.  All
    sub-configs carry sensible defaults so that ``VisualizationConfig()`` yields
    a production-ready dark-scientific theme.

    Parameters
    ----------
    colors : ColorPalette
        Colour definitions for the entire visualisation.
    render : RenderConfig
        Rendering and canvas layout parameters.
    projection : ProjectionConfig
        Dimensionality-reduction settings applied to model inputs and embeddings.
    title : str, optional
        Plot title string.  When set, overrides the auto-generated title.
    show_variance_annotation : bool
        When ``True`` and PCA is used, annotates the figure with the cumulative
        explained-variance ratio captured by the three principal components.

    Examples
    --------
    >>> from geolatent.config.themes import VisualizationConfig, ProjectionConfig
    >>> cfg = VisualizationConfig()
    >>> cfg_tsne = cfg.with_method("tsne")
    >>> cfg_titled = cfg.with_title("SVM Decision Geometry — RBF Kernel")
    """

    colors: ColorPalette = field(default_factory=ColorPalette)
    render: RenderConfig = field(default_factory=RenderConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    title: Optional[str] = None
    show_variance_annotation: bool = True

    # Fluent mutation helpers (return new copies to avoid shared state)

    def copy(self) -> "VisualizationConfig":
        """Return an independent deep copy of this configuration object."""
        return copy.deepcopy(self)

    def with_method(self, method: str) -> "VisualizationConfig":
        """Return a copy with the projection ``method`` overridden.

        Parameters
        ----------
        method : {"pca", "tsne", "umap", "sensitivity"}
            New projection algorithm.

        Returns
        -------
        VisualizationConfig
        """
        cfg = self.copy()
        cfg.projection.method = method
        return cfg

    def with_title(self, title: str) -> "VisualizationConfig":
        """Return a copy with the plot ``title`` overridden.

        Parameters
        ----------
        title : str

        Returns
        -------
        VisualizationConfig
        """
        cfg = self.copy()
        cfg.title = title
        return cfg

    def with_resolution(self, width: int, height: int) -> "VisualizationConfig":
        """Return a copy with canvas dimensions overridden.

        Parameters
        ----------
        width : int
        height : int

        Returns
        -------
        VisualizationConfig
        """
        cfg = self.copy()
        cfg.render.width = width
        cfg.render.height = height
        return cfg

    def with_opacity(
        self,
        surface: Optional[float] = None,
        scatter: Optional[float] = None,
        ellipsoid: Optional[float] = None,
    ) -> "VisualizationConfig":
        """Return a copy with opacity values selectively overridden.

        Parameters
        ----------
        surface : float, optional
        scatter : float, optional
        ellipsoid : float, optional

        Returns
        -------
        VisualizationConfig
        """
        cfg = self.copy()
        if surface is not None:
            cfg.render.surface_opacity = surface
        if scatter is not None:
            cfg.render.scatter_opacity = scatter
        if ellipsoid is not None:
            cfg.render.ellipsoid_opacity = ellipsoid
        return cfg


# Pre-built themes

#: Default dark-scientific theme shipped with geolatent.
#: Colour palette inspired by GitHub Dark, Weights & Biases, and the visual
#: language of high-quality ML research tooling.
DARK_SCIENTIFIC: VisualizationConfig = VisualizationConfig()

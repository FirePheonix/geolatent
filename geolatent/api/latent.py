"""High-level API: ``inspect_latent_space``.

This module exposes the primary entry point for embedding / latent-space analysis.
Unlike :func:`~geolatent.api.decision.visualize_decision_geometry`, which
renders model decision boundaries, ``inspect_latent_space`` focuses on the
*geometric structure of the representation itself*: how well-separated are
class clusters, what is the intrinsic dimensionality of the manifold, and do
subgroups form coherent neighbourhoods?

The function supports arbitrary high-dimensional embeddings — word vectors,
image feature maps, VAE latent codes, transformer hidden states, etc. — and
reduces them to 3 principal components (or t-SNE / UMAP coordinates) before
rendering.

Usage
-----
::

    from geolatent import inspect_latent_space

    # BERT sentence embeddings, shape (512, 768)
    fig = inspect_latent_space(
        embeddings=bert_output,
        labels=sentence_classes,
        projection_method="tsne",
        title="BERT Sentence Embeddings — Topic Clusters",
    )
    fig.show()
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

from ..config.themes import DARK_SCIENTIFIC, VisualizationConfig
from ..core.projector import DimensionalityProjector
from ..rendering.overlays import DataOverlay
from ..rendering.scene import Scene3D
from ..utils.validators import (
    validate_class_names,
    validate_embeddings,
)


def inspect_latent_space(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    config: Optional[VisualizationConfig] = None,
    projection_method: str = "pca",
    show_scatter: bool = True,
    show_centroids: bool = True,
    show_ellipsoids: bool = True,
    show_convex_hulls: bool = False,
    ellipsoid_confidence: float = 0.90,
    class_names: Optional[Dict] = None,
    title: Optional[str] = None,
    point_size: Optional[int] = None,
    scatter_opacity: Optional[float] = None,
) -> go.Figure:
    """Visualise the geometric structure of high-dimensional embeddings in 3-D.

    Projects *embeddings* from their native dimensionality to 3-D using the
    chosen dimensionality-reduction algorithm, then renders an interactive scene
    with class-coloured scatter clouds, centroid markers, and optional structural
    overlays (confidence ellipsoids, convex hulls).

    Parameters
    ----------
    embeddings : array-like of shape (n_samples, n_dims)
        High-dimensional embedding vectors.  Suitable inputs include
        transformer hidden states, GAN latent codes, learned feature maps, or
        any continuous representation to be analysed structurally.
    labels : array-like of shape (n_samples,)
        Integer or string class / group labels for colour coding.
    config : VisualizationConfig, optional
        Theme and rendering configuration.  Defaults to ``DARK_SCIENTIFIC``.
    projection_method : {"pca", "tsne", "umap"}
        Dimensionality-reduction algorithm.  ``"pca"`` is fast and preserves
        global geometry; ``"tsne"`` / ``"umap"`` are better at revealing
        local cluster structure at the cost of interpretability.
    show_scatter : bool
        Whether to render the data-point scatter cloud.
    show_centroids : bool
        Whether to render class-centroid diamond markers.
    show_ellipsoids : bool
        Whether to overlay Mahalanobis-distance confidence ellipsoids (default
        ``True`` — these are the primary structural indicator in latent-space
        analysis).
    show_convex_hulls : bool
        Whether to overlay transparent convex-hull surfaces per class.
    ellipsoid_confidence : float
        Confidence level for the Mahalanobis ellipsoid (default 0.90).
    class_names : dict, optional
        Mapping from label value to display string.
    title : str, optional
        Figure title.
    point_size : int, optional
        Override the default scatter marker size.
    scatter_opacity : float, optional
        Override the default scatter marker opacity.

    Returns
    -------
    fig : plotly.graph_objects.Figure

    Raises
    ------
    ValueError
        If ``embeddings`` or ``labels`` fail shape / finiteness validation.

    Examples
    --------
    >>> from geolatent import inspect_latent_space
    >>> import numpy as np
    >>>
    >>> # Simulate 4 Gaussian clusters in a 128-D embedding space
    >>> rng = np.random.default_rng(0)
    >>> embeddings = np.vstack([
    ...     rng.normal(loc=c, scale=1.0, size=(100, 128))
    ...     for c in [0, 3, 6, 9]
    ... ])
    >>> labels = np.repeat([0, 1, 2, 3], 100)
    >>> fig = inspect_latent_space(
    ...     embeddings, labels,
    ...     projection_method="pca",
    ...     title="128-D Gaussian Clusters — PCA Projection",
    ... )
    >>> fig.show()
    """
    # 1. Validate
    embeddings, labels = validate_embeddings(embeddings, labels)
    class_names = validate_class_names(class_names, np.unique(labels))

    # 2. Configuration
    cfg = (config or DARK_SCIENTIFIC).copy()
    cfg.projection.method = projection_method
    if title:
        cfg.title = title

    # 3. Project to 3-D
    projector = DimensionalityProjector(cfg.projection)
    proj_result = projector.fit_transform(embeddings)
    E_3d = proj_result.coordinates

    # 4. Scene assembly
    scene = Scene3D(cfg)
    scene.set_axis_labels(proj_result.axis_labels)

    overlay = DataOverlay(cfg)

    # 5. Scatter
    if show_scatter:
        scatter_traces = overlay.render_scatter(
            E_3d,
            labels,
            class_names=class_names,
            point_size_override=point_size,
            opacity_override=scatter_opacity,
        )
        scene.add_traces(scatter_traces)

    # 6. Confidence ellipsoids
    if show_ellipsoids:
        ellipsoid_traces = overlay.render_ellipsoids(
            E_3d,
            labels,
            confidence=ellipsoid_confidence,
            class_names=class_names,
        )
        scene.add_traces(ellipsoid_traces)

    # 7. Convex hulls
    if show_convex_hulls:
        hull_traces = overlay.render_convex_hulls(E_3d, labels, class_names=class_names)
        scene.add_traces(hull_traces)

    # 8. Class centroids
    if show_centroids:
        centroid_trace = overlay.render_centroids(
            E_3d, labels, class_names=class_names
        )
        scene.add_trace(centroid_trace)

    # 9. Annotations
    final_title = cfg.title or _auto_title(embeddings, projection_method)
    scene.set_title(final_title)

    if cfg.show_variance_annotation and proj_result.explained_variance_ratio is not None:
        scene.add_variance_annotation(proj_result.explained_variance_ratio)

    n_samples, n_dims = embeddings.shape
    n_classes = len(np.unique(labels))
    scene.add_text_annotation(
        f"n={n_samples:,}  dim={n_dims}-&gt;3  "
        f"{n_classes} classes  {projection_method.upper()}",
        x=0.99,
        y=0.99,
    )

    return scene.render()


# Private helpers


def _auto_title(embeddings: np.ndarray, projection_method: str) -> str:
    n, d = embeddings.shape
    proj_tag = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}.get(
        projection_method, projection_method.upper()
    )
    return f"Latent Space — {d}-D embeddings ({proj_tag} projection)"

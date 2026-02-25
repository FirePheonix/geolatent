"""High-level API: ``visualize_decision_geometry``.

This module exposes the primary entry point for decision-boundary analysis.
The function orchestrates the full pipeline:

1. **Validation** — coerce and validate inputs; surface informative errors early.
2. **Projection** — fit a :class:`~geolatent.core.projector.DimensionalityProjector`
   (PCA by default) on the training data ``X``, reduce to 3 principal components.
3. **Mesh construction** — when PCA is used, build a regular 3-D prediction mesh
   by querying the model on a grid of inverse-transformed points.
4. **Scene assembly** — instantiate :class:`~geolatent.rendering.scene.Scene3D`,
   layer decision surfaces, scatter clouds, centroids, and optional structural
   overlays according to the caller-supplied flags.
5. **Rendering** — apply the dark-scientific theme and return the completed
   ``go.Figure``.

Usage
-----
::

    from geolatent import visualize_decision_geometry
    from sklearn.svm import SVC

    model = SVC(kernel="rbf", probability=True).fit(X_train, y_train)
    fig = visualize_decision_geometry(model, X_train, y_train)
    fig.show()
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

from ..config.themes import DARK_SCIENTIFIC, VisualizationConfig
from ..core.geometry import GeometryUtils
from ..core.mesh_builder import MeshBuilder
from ..core.projector import DimensionalityProjector
from ..rendering.overlays import DataOverlay
from ..rendering.scene import Scene3D
from ..rendering.surfaces import DecisionSurfaceRenderer
from ..utils.validators import (
    validate_class_names,
    validate_classification_labels,
    validate_feature_matrix,
    validate_label_vector,
    validate_sklearn_model,
)


def visualize_decision_geometry(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    *,
    config: Optional[VisualizationConfig] = None,
    projection_method: str = "pca",
    predict_fn=None,
    feature_names: Optional[List[str]] = None,
    mesh_resolution: int = 30,
    show_surface: bool = True,
    show_confidence: bool = True,
    show_scatter: bool = True,
    show_centroids: bool = True,
    show_ellipsoids: bool = False,
    show_convex_hulls: bool = False,
    ellipsoid_confidence: float = 0.90,
    class_names: Optional[Dict] = None,
    title: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> go.Figure:
    """Render the decision geometry of a scikit-learn-compatible classifier in 3-D.

    The input feature matrix ``X`` is projected to 3 principal components via
    PCA (or t-SNE / UMAP for pure scatter visualisation).  When PCA is used,
    the model's decision function is evaluated on a regular 3-D grid that is
    inverse-transformed back into the original feature space, producing decision
    boundary isosurfaces anchored to the actual model geometry — not an
    approximation in an arbitrary slice.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Must implement ``predict(X)``.  Also implements ``predict_proba(X)``
        for richer confidence-surface rendering (recommended).
    X : array-like of shape (n_samples, n_features)
        Training feature matrix.  Will be standardised and projected internally.
    y : array-like of shape (n_samples,)
        Class label vector.  Integer or string labels are both supported.
    config : VisualizationConfig, optional
        Custom theme and rendering configuration.  Defaults to
        :data:`~geolatent.config.themes.DARK_SCIENTIFIC`.
    projection_method : {"pca", "tsne", "umap", "sensitivity"}
        Dimensionality-reduction algorithm.  ``"pca"`` and ``"sensitivity"``
        both support decision-surface rendering.  ``"sensitivity"`` uses
        finite-difference Jacobians to find axes the model actually cares about
        and works with any callable (sklearn, PyTorch, XGBoost, etc.).
    predict_fn : callable, optional
        Required when ``projection_method="sensitivity"`` with a non-sklearn
        model.  For sklearn models it is auto-derived from ``model.predict_proba``
        or ``model.predict`` when not supplied.
    feature_names : list of str, optional
        Names of the input features.  Shown on axes and sensitivity labels.
    mesh_resolution : int
        Grid resolution per dimension for the prediction mesh.  Total inference
        calls equal ``mesh_resolution³``.  Default 30.
    show_surface : bool
        Whether to render the decision boundary / probability surfaces.
    show_confidence : bool
        When ``True`` and the model exposes ``predict_proba``, render nested
        confidence isosurfaces in addition to the primary boundary shell.
    show_scatter : bool
        Whether to render the data-point scatter cloud.
    show_centroids : bool
        Whether to render class-centroid diamond markers.
    show_ellipsoids : bool
        Whether to overlay Mahalanobis-distance confidence ellipsoids.
    show_convex_hulls : bool
        Whether to overlay transparent convex-hull surfaces per class.
    ellipsoid_confidence : float
        Confidence level for ellipsoid construction (default 0.90 → 90 % region).
    class_names : dict, optional
        Mapping from class label to human-readable display string.
    title : str, optional
        Figure title.  Overrides ``config.title`` when supplied.
    batch_size : int, optional
        Batch size for model inference on the prediction mesh.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive 3-D Plotly figure.

    Raises
    ------
    TypeError
        If ``model`` does not expose a ``predict`` method.
    ValueError
        If ``X`` or ``y`` fail validation (shape, NaN, insufficient classes).

    Examples
    --------
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> from sklearn.datasets import make_classification
    >>> from geolatent import visualize_decision_geometry
    >>>
    >>> X, y = make_classification(n_samples=400, n_features=20, n_classes=3,
    ...                            n_informative=10, random_state=0)
    >>> clf = GradientBoostingClassifier(n_estimators=50, random_state=0).fit(X, y)
    >>> fig = visualize_decision_geometry(clf, X, y,
    ...                                   title="GBM — 3-class Decision Geometry")
    >>> fig.show()
    """
    # 1. Validate inputs
    validate_sklearn_model(model)
    X = validate_feature_matrix(X, min_samples=4, min_features=2)
    y = validate_label_vector(y, n_samples=len(X))
    y = validate_classification_labels(y)
    class_names = validate_class_names(class_names, np.unique(y))

    # 2. Resolve configuration
    cfg = (config or DARK_SCIENTIFIC).copy()
    cfg.projection.method = projection_method
    if title:
        cfg.title = title

    invertible_methods = ("pca", "sensitivity")
    if projection_method not in invertible_methods and show_surface:
        warnings.warn(
            f"projection_method='{projection_method}' does not support "
            "inverse_transform; decision surfaces will not be rendered.  "
            "Use projection_method='pca' or 'sensitivity'.",
            UserWarning,
            stacklevel=2,
        )
        show_surface = False

    # 3. Auto-derive predict_fn for sensitivity when not supplied
    if projection_method == "sensitivity" and predict_fn is None:
        if hasattr(model, "predict_proba"):
            predict_fn = model.predict_proba
        else:
            predict_fn = lambda X_: model.predict(X_).astype(np.float64)  # noqa: E731

    # 4. Project data to 3-D
    projector = DimensionalityProjector(cfg.projection)
    proj_result = projector.fit_transform(X, predict_fn=predict_fn, feature_names=feature_names)
    X_3d = proj_result.coordinates

    # 5. Build prediction mesh
    mesh = None
    if show_surface and projector.supports_inverse_transform:
        try:
            builder = MeshBuilder(
                resolution=mesh_resolution,
                batch_size=batch_size,
            )
            mesh = builder.build_prediction_mesh(model, projector, X_3d)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Mesh construction failed: {exc!r}.  "
                "Proceeding without decision surfaces.",
                UserWarning,
                stacklevel=2,
            )

    # 5. Assemble scene
    scene = Scene3D(cfg)
    scene.set_axis_labels(proj_result.axis_labels)

    # 6. Decision surfaces
    if mesh is not None:
        renderer = DecisionSurfaceRenderer(cfg)
        surface_traces = renderer.render(
            mesh,
            class_names=class_names,
            show_confidence=show_confidence,
        )
        scene.add_traces(surface_traces)

    # 7. Data-point scatter
    overlay = DataOverlay(cfg)

    if show_scatter:
        scatter_traces = overlay.render_scatter(X_3d, y, class_names=class_names)
        scene.add_traces(scatter_traces)

    # 8. Class centroids
    if show_centroids:
        centroid_trace = overlay.render_centroids(X_3d, y, class_names=class_names)
        scene.add_trace(centroid_trace)

    # 9. Confidence ellipsoids
    if show_ellipsoids:
        ellipsoid_traces = overlay.render_ellipsoids(
            X_3d,
            y,
            confidence=ellipsoid_confidence,
            class_names=class_names,
        )
        scene.add_traces(ellipsoid_traces)

    # 10. Convex hulls
    if show_convex_hulls:
        hull_traces = overlay.render_convex_hulls(X_3d, y, class_names=class_names)
        scene.add_traces(hull_traces)

    # 11. Annotations
    final_title = cfg.title or _auto_title(model, projection_method)
    scene.set_title(final_title)

    if cfg.show_variance_annotation and proj_result.explained_variance_ratio is not None:
        scene.add_variance_annotation(proj_result.explained_variance_ratio)

    n_samples, n_features = X.shape
    scene.add_text_annotation(
        f"n={n_samples:,}  d={n_features}-&gt;3  "
        f"{type(model).__name__}",
        x=0.99,
        y=0.99,
    )

    return scene.render()


# Private helpers


def _auto_title(model: object, projection_method: str) -> str:
    model_name = type(model).__name__
    proj_tag = {
        "pca": "PCA", "tsne": "t-SNE", "umap": "UMAP", "sensitivity": "Sensitivity"
    }.get(projection_method, projection_method.upper())
    return f"Decision Geometry — {model_name} ({proj_tag} projection)"

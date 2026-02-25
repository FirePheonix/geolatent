"""Dimensionality-reduction pipeline for geolatent.

Central class: :class:`DimensionalityProjector`.

Design principles
-----------------
* The projector acts as a stateful transform pipeline (fit → transform)
  following the scikit-learn estimator convention, which keeps it composable
  with existing ML pipelines.

* Only PCA guarantees an analytically invertible linear transform.  t-SNE and
  UMAP projections are therefore restricted to latent-space visualisation;
  requesting a decision surface with these methods raises a clear error rather
  than silently producing a misleading result.

* Input standardisation (zero-mean, unit-variance) is applied internally when
  ``ProjectionConfig.scale_input`` is ``True``, ensuring numerical stability
  across all supported algorithms and preventing features with large dynamic
  range from dominating the projection.

* The projector stores the fitted scaler alongside the projection model so that
  the full round-trip  X → (scale) → (project) → X_3d → (unproject) → (unscale) → X̂
  is encapsulated in a single object.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..config.themes import ProjectionConfig


# Result container


@dataclass
class ProjectionResult:
    """Container returned by :meth:`DimensionalityProjector.fit_transform`.

    Attributes
    ----------
    coordinates : np.ndarray of shape (n_samples, 3)
        3-D projected coordinates for each input sample.
    explained_variance_ratio : np.ndarray of shape (3,) or None
        Fraction of total variance captured by each principal component.
        ``None`` when a non-PCA method is used.
    cumulative_variance : float or None
        Sum of ``explained_variance_ratio``; convenience scalar indicating how
        much information is preserved in the projection.
    method : str
        Name of the projection algorithm that produced these coordinates.
    original_dim : int
        Number of features in the original (un-projected) space.
    axis_labels : list of str
        Suggested axis labels for the three projected dimensions.
    """

    coordinates: np.ndarray
    explained_variance_ratio: Optional[np.ndarray]
    cumulative_variance: Optional[float]
    method: str
    original_dim: int
    axis_labels: List[str] = field(default_factory=lambda: ["PC 1", "PC 2", "PC 3"])


# Main projector class


class DimensionalityProjector:
    """Unified dimensionality-reduction pipeline.

    Supports PCA (with invertible transform), t-SNE, and UMAP.  The projector
    follows the fit/transform convention so it can be reused across multiple
    datasets sharing the same embedding geometry — e.g., projecting a test set
    using the PCA basis fitted on training data.

    Parameters
    ----------
    config : ProjectionConfig
        Projection hyper-parameters and algorithm selection.

    Attributes
    ----------
    supports_inverse_transform : bool
        ``True`` when the underlying algorithm provides a meaningful inverse
        mapping from the low-dimensional space back to the original feature
        space.  Currently ``True`` only for PCA and the identity projection
        (when the input is already ≤ 3-dimensional).
    is_fitted : bool
        ``True`` after :meth:`fit` or :meth:`fit_transform` has been called.

    Examples
    --------
    >>> from geolatent.config.themes import ProjectionConfig
    >>> from geolatent.core.projector import DimensionalityProjector
    >>> import numpy as np
    >>> cfg = ProjectionConfig(method="pca", scale_input=True)
    >>> proj = DimensionalityProjector(cfg)
    >>> X = np.random.randn(300, 50)
    >>> result = proj.fit_transform(X)
    >>> result.coordinates.shape
    (300, 3)
    >>> result.cumulative_variance  # doctest: +SKIP
    0.312...
    """

    def __init__(self, config: ProjectionConfig) -> None:
        self.config = config
        self._projector = None
        self._scaler: Optional[StandardScaler] = None
        self._is_fitted: bool = False
        self._used_identity_path: bool = False  # True when n_features <= 3
        self._is_sensitivity: bool = False       # True when method == "sensitivity"
        self._sens_components: Optional[np.ndarray] = None  # (n_components, n_features)
        self._sens_X_mean: Optional[np.ndarray] = None       # (n_features,)
        self._feature_names: Optional[List[str]] = None
        self.supports_inverse_transform: bool = False
        self.is_fitted: bool = False

    # Public interface

    def fit(self, X: np.ndarray) -> "DimensionalityProjector":
        """Fit the projection on training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.  Will be standardised internally if
            ``config.scale_input`` is ``True``.

        Returns
        -------
        self : DimensionalityProjector
            The fitted projector instance, supporting method chaining.
        """
        X = self._coerce_input(X)
        n_features = X.shape[1]
        if n_features <= 3:
            # trivial identity path — fit scaler only
            self._fit_scaler(X)
            self.supports_inverse_transform = True
        else:
            X_scaled = self._fit_scaler(X)
            self._fit_projector(X_scaled)
        self._is_fitted = True
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project *X* using the already-fitted transform.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix; must have the same number of features as the
            data used for fitting.

        Returns
        -------
        X_proj : np.ndarray of shape (n_samples, 3)
            3-D projected coordinates.
        """
        self._assert_fitted()
        X = self._coerce_input(X)
        X_scaled = self._apply_scaler(X)
        if X.shape[1] <= 3:
            return self._handle_low_dim(X_scaled, X.shape[1])
        return self._apply_projector(X_scaled)

    def fit_transform(
        self,
        X: np.ndarray,
        predict_fn=None,
        feature_names: Optional[List[str]] = None,
    ) -> "ProjectionResult":
        """Fit the projection and immediately transform the input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.
        predict_fn : callable, optional
            Required when ``config.method == "sensitivity"``.  Any callable
            that accepts an ``(n, n_features)`` array and returns predictions
            of shape ``(n,)`` or ``(n, n_classes)``.  Works with sklearn,
            PyTorch, XGBoost, or any custom function.
        feature_names : list of str, optional
            Names of the input features, used for axis labels.

        Returns
        -------
        result : ProjectionResult
            Projected coordinates with associated diagnostic metadata.
        """
        X = self._coerce_input(X)
        n_samples, n_features = X.shape
        self._feature_names = list(feature_names) if feature_names is not None else None

        # fast path: data already lives in ≤ 3 dimensions
        if n_features <= 3 and self.config.method not in ("tsne", "umap"):
            X_scaled = self._fit_scaler(X)
            coords = self._handle_low_dim(X_scaled, n_features)
            self.supports_inverse_transform = True
            self._used_identity_path = True
            self._is_fitted = True
            self.is_fitted = True
            labels = (
                list(feature_names[:n_features]) + ["—"] * (3 - n_features)
                if feature_names is not None
                else self._axis_labels("identity", n_features)
            )
            return ProjectionResult(
                coordinates=coords,
                explained_variance_ratio=None,
                cumulative_variance=None,
                method="identity",
                original_dim=n_features,
                axis_labels=labels,
            )

        # general path
        X_scaled = self._fit_scaler(X)

        if self.config.method.lower() == "sensitivity":
            if predict_fn is None:
                raise ValueError(
                    "predict_fn is required for projection_method='sensitivity'. "
                    "Pass any callable: model.predict_proba, a PyTorch forward "
                    "function, etc."
                )
            self._fit_sensitivity_projector(X_scaled, predict_fn)
            coords = self._apply_projector(X_scaled)
        else:
            self._fit_projector(X_scaled)
            coords = self._apply_projector(X_scaled)

        self._is_fitted = True
        self.is_fitted = True

        ev_ratio: Optional[np.ndarray] = None
        cum_var: Optional[float] = None
        if (
            not self._is_sensitivity
            and hasattr(self._projector, "explained_variance_ratio_")
        ):
            ev_ratio = self._projector.explained_variance_ratio_.copy()
            cum_var = float(np.sum(ev_ratio))

        return ProjectionResult(
            coordinates=coords,
            explained_variance_ratio=ev_ratio,
            cumulative_variance=cum_var,
            method=self.config.method,
            original_dim=n_features,
            axis_labels=self._axis_labels(self.config.method, n_features),
        )

    def inverse_transform(self, X_proj: np.ndarray) -> np.ndarray:
        """Map projected coordinates back to the original feature space.

        This method is only meaningful for PCA and the identity projection.
        For t-SNE / UMAP, it raises ``NotImplementedError``.

        Parameters
        ----------
        X_proj : np.ndarray of shape (n_points, 3)
            Points in the 3-D projected space.

        Returns
        -------
        X_original : np.ndarray of shape (n_points, n_features)
            Best-rank-3 approximation of the corresponding feature vectors in
            the original (possibly high-dimensional) feature space.

        Raises
        ------
        NotImplementedError
            When the projector was fitted with t-SNE or UMAP.
        RuntimeError
            When the projector has not been fitted yet.
        """
        if not self.supports_inverse_transform:
            raise NotImplementedError(
                f"The '{self.config.method}' projection does not provide an "
                "inverse transform.  Decision-surface rendering requires the "
                "PCA projection method."
            )
        self._assert_fitted()

        # Identity path: data was already <= 3-D, unscale only
        if self._used_identity_path:
            if self._scaler is not None:
                return self._scaler.inverse_transform(X_proj)
            return X_proj.copy()

        # Sensitivity path: linear reconstruction via stored components
        if self._is_sensitivity:
            X_scaled_rec = X_proj @ self._sens_components + self._sens_X_mean
            if self._scaler is not None:
                return self._scaler.inverse_transform(X_scaled_rec)
            return X_scaled_rec

        # PCA path
        X_scaled_rec = self._projector.inverse_transform(X_proj)
        if self._scaler is not None:
            return self._scaler.inverse_transform(X_scaled_rec)
        return X_scaled_rec

    # Private helpers

    @staticmethod
    def _coerce_input(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(
                f"Input must be 2-D with shape (n_samples, n_features); "
                f"got shape {X.shape}."
            )
        return X

    def _fit_scaler(self, X: np.ndarray) -> np.ndarray:
        if not self.config.scale_input:
            return X
        self._scaler = StandardScaler()
        return self._scaler.fit_transform(X)

    def _apply_scaler(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is None:
            return X
        return self._scaler.transform(X)

    def _fit_projector(self, X_scaled: np.ndarray) -> None:
        method = self.config.method.lower()
        n_components = min(
            self.config.n_components,
            X_scaled.shape[1],
            X_scaled.shape[0] - 1,
        )

        if method == "pca":
            self._projector = PCA(
                n_components=n_components,
                whiten=self.config.whiten,
                random_state=self.config.random_state,
            )
            self._projector.fit(X_scaled)
            self.supports_inverse_transform = True

        elif method == "tsne":
            from sklearn.manifold import TSNE

            n_samples = X_scaled.shape[0]
            perplexity = min(self.config.tsne_perplexity, n_samples - 1)
            if perplexity != self.config.tsne_perplexity:
                warnings.warn(
                    f"t-SNE perplexity reduced from {self.config.tsne_perplexity} "
                    f"to {perplexity} to satisfy n_samples constraint.",
                    UserWarning,
                    stacklevel=4,
                )
            import sklearn
            _sklearn_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
            _iter_kwarg = "max_iter" if _sklearn_ver >= (1, 5) else "n_iter"
            self._projector = TSNE(
                n_components=n_components,
                perplexity=float(perplexity),
                learning_rate=self.config.tsne_learning_rate,
                **{_iter_kwarg: self.config.tsne_n_iter},
                random_state=self.config.random_state,
                init="pca",
            )
            self.supports_inverse_transform = False
            # Store scaled data for use in fit_transform (t-SNE cannot separate fit/transform)
            self._tsne_train_data = X_scaled

        elif method == "umap":
            try:
                import umap as umap_module  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "The 'umap-learn' package is not installed.  "
                    "Install it with:  pip install umap-learn"
                ) from exc
            self._projector = umap_module.UMAP(
                n_components=n_components,
                n_neighbors=self.config.umap_n_neighbors,
                min_dist=self.config.umap_min_dist,
                random_state=self.config.random_state,
            )
            self._projector.fit(X_scaled)
            self.supports_inverse_transform = False

        else:
            raise ValueError(
                f"Unknown projection method '{method}'.  "
                "Choose from: 'pca', 'tsne', 'umap', 'sensitivity'."
            )

    def _apply_projector(self, X_scaled: np.ndarray) -> np.ndarray:
        method = self.config.method.lower()
        if self._is_sensitivity:
            # Linear projection: centre on X mean, project onto sensitivity components
            return (X_scaled - self._sens_X_mean) @ self._sens_components.T
        if method == "tsne":
            return self._projector.fit_transform(X_scaled)
        return self._projector.transform(X_scaled)

    def _fit_sensitivity_projector(self, X_scaled: np.ndarray, predict_fn) -> None:
        """Fit a model-driven linear projection via finite-difference Jacobians.

        For each feature j, perturbs X by eps along that axis and measures
        how much the model output changes.  PCA on the resulting sensitivity
        matrix finds directions in feature space where the model's decisions
        vary most — regardless of data variance.  The projection is linear
        and invertible, so decision surfaces still work.
        """
        n_samples, n_features = X_scaled.shape
        n_sub = min(n_samples, self.config.sensitivity_n_subsample)
        eps = self.config.sensitivity_eps

        rng = np.random.default_rng(self.config.random_state)
        idx = rng.choice(n_samples, n_sub, replace=False)
        X_sub = X_scaled[idx]

        # Base predictions — support any output shape
        base = np.asarray(predict_fn(X_sub), dtype=np.float64)
        if base.ndim == 1:
            base = base[:, None]

        # Sensitivity matrix S[i, j] = mean |doutput/dfeature_j| at sample i
        S = np.zeros((n_sub, n_features), dtype=np.float64)
        for j in range(n_features):
            X_pert = X_sub.copy()
            X_pert[:, j] += eps
            pert = np.asarray(predict_fn(X_pert), dtype=np.float64)
            if pert.ndim == 1:
                pert = pert[:, None]
            S[:, j] = np.abs(pert - base).mean(axis=1) / eps

        # PCA on S finds the axes of maximum sensitivity variation in feature space
        n_components = min(self.config.n_components, n_features, n_sub - 1)
        sens_pca = PCA(n_components=n_components, random_state=self.config.random_state)
        sens_pca.fit(S)

        self._sens_components = sens_pca.components_          # (n_components, n_features)
        self._sens_X_mean = X_scaled.mean(axis=0)             # (n_features,)
        self._is_sensitivity = True
        self.supports_inverse_transform = True

    @staticmethod
    def _handle_low_dim(X_scaled: np.ndarray, n_features: int) -> np.ndarray:
        """Pad 1-D or 2-D inputs to 3 columns with zeros."""
        if n_features == 3:
            return X_scaled.copy()
        n_samples = X_scaled.shape[0]
        pad = np.zeros((n_samples, 3 - n_features), dtype=np.float64)
        return np.concatenate([X_scaled, pad], axis=1)

    def _axis_labels(self, method: str, original_dim: int) -> List[str]:
        method = method.lower()
        if method in ("pca", "identity"):
            if original_dim <= 3:
                if self._feature_names:
                    labels = list(self._feature_names[:original_dim])
                else:
                    labels = [f"Feature {i + 1}" for i in range(original_dim)]
                while len(labels) < 3:
                    labels.append("—")
                return labels
            return ["PC 1", "PC 2", "PC 3"]
        if method == "sensitivity":
            return self._sensitivity_axis_labels()
        if method == "tsne":
            return ["t-SNE 1", "t-SNE 2", "t-SNE 3"]
        if method == "umap":
            return ["UMAP 1", "UMAP 2", "UMAP 3"]
        return ["Dim 1", "Dim 2", "Dim 3"]

    def _sensitivity_axis_labels(self) -> List[str]:
        """Generate axis labels showing top contributing features per sensitivity axis."""
        if self._sens_components is None:
            return ["Sens 1", "Sens 2", "Sens 3"]
        labels = []
        for i, comp in enumerate(self._sens_components):
            top_idx = np.argsort(np.abs(comp))[::-1][:2]
            if self._feature_names:
                top_names = [self._feature_names[j] for j in top_idx]
                label = f"Sens {i+1} ({', '.join(top_names)})"
            else:
                label = f"Sens {i+1} (feat {', '.join(str(j) for j in top_idx)})"
            labels.append(label)
        return labels

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "DimensionalityProjector is not fitted.  "
                "Call fit() or fit_transform() first."
            )

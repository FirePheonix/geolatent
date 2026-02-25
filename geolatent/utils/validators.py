"""Input validation utilities for geolatent.

All public API functions route their inputs through the validators defined here
before any computation begins.  Validation is strict-but-informative: errors
surface with enough context to enable fast diagnosis without requiring the user
to read source code.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


# Data validators


def validate_feature_matrix(
    X: Any,
    *,
    min_samples: int = 4,
    min_features: int = 2,
    name: str = "X",
) -> np.ndarray:
    """Validate and coerce a feature matrix.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input feature matrix.
    min_samples : int
        Minimum acceptable number of samples.
    min_features : int
        Minimum acceptable number of features.
    name : str
        Variable name used in error messages.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features), dtype float64

    Raises
    ------
    TypeError
        If *X* cannot be converted to a NumPy array.
    ValueError
        If *X* is not 2-D, contains fewer than *min_samples* rows,
        fewer than *min_features* columns, or contains non-finite values.
    """
    try:
        X = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"'{name}' must be convertible to a float64 NumPy array; "
            f"got type {type(X).__name__}."
        ) from exc

    if X.ndim != 2:
        raise ValueError(
            f"'{name}' must be a 2-D array of shape (n_samples, n_features), "
            f"got shape {X.shape}."
        )

    n_samples, n_features = X.shape

    if n_samples < min_samples:
        raise ValueError(
            f"'{name}' must have at least {min_samples} samples; "
            f"got {n_samples}."
        )

    if n_features < min_features:
        raise ValueError(
            f"'{name}' must have at least {min_features} features for 3-D "
            f"projection; got {n_features}."
        )

    if not np.isfinite(X).all():
        n_nan = np.sum(np.isnan(X))
        n_inf = np.sum(np.isinf(X))
        raise ValueError(
            f"'{name}' contains {n_nan} NaN value(s) and {n_inf} Inf value(s). "
            "Impute or remove these before calling geolatent."
        )

    return X


def validate_label_vector(
    y: Any,
    *,
    n_samples: int,
    name: str = "y",
) -> np.ndarray:
    """Validate and coerce a label / target vector.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Class labels or regression targets.
    n_samples : int
        Expected length (must match the corresponding feature matrix).
    name : str
        Variable name used in error messages.

    Returns
    -------
    y : np.ndarray of shape (n_samples,)

    Raises
    ------
    TypeError
        If *y* cannot be converted to a NumPy array.
    ValueError
        If *y* is not 1-D or its length does not match *n_samples*.
    """
    try:
        y = np.asarray(y)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"'{name}' must be convertible to a NumPy array; "
            f"got type {type(y).__name__}."
        ) from exc

    if y.ndim != 1:
        raise ValueError(
            f"'{name}' must be a 1-D array of shape (n_samples,), "
            f"got shape {y.shape}."
        )

    if len(y) != n_samples:
        raise ValueError(
            f"Length mismatch: X has {n_samples} samples but '{name}' has "
            f"{len(y)} entries."
        )

    return y


def validate_classification_labels(y: np.ndarray, *, name: str = "y") -> np.ndarray:
    """Assert that *y* is suitable for classification.

    Parameters
    ----------
    y : np.ndarray
        Label vector (already validated by :func:`validate_label_vector`).
    name : str

    Returns
    -------
    y : np.ndarray
        The input unchanged.

    Raises
    ------
    ValueError
        If *y* contains only one unique class (degenerate problem).
    """
    n_classes = len(np.unique(y))
    if n_classes < 2:
        raise ValueError(
            f"'{name}' must contain at least 2 distinct classes for "
            f"classification visualisation; found {n_classes}."
        )
    return y


def validate_embeddings(
    embeddings: Any,
    labels: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate an (embeddings, labels) pair for latent-space visualisation.

    Parameters
    ----------
    embeddings : array-like of shape (n_samples, n_dims)
        High-dimensional embedding vectors.
    labels : array-like of shape (n_samples,)
        Integer or string class labels.

    Returns
    -------
    embeddings : np.ndarray of shape (n_samples, n_dims)
    labels : np.ndarray of shape (n_samples,)
    """
    embeddings = validate_feature_matrix(embeddings, min_features=2, name="embeddings")
    labels = validate_label_vector(labels, n_samples=len(embeddings), name="labels")
    return embeddings, labels


# Model validators


def validate_sklearn_model(model: Any, *, require_predict_proba: bool = False) -> None:
    """Validate that *model* exposes a scikit-learn-compatible interface.

    Parameters
    ----------
    model : Any
        The model to validate.
    require_predict_proba : bool
        If ``True``, also assert that ``predict_proba`` is present.

    Raises
    ------
    TypeError
        If *model* does not expose ``predict``.
    AttributeError
        If *require_predict_proba* is ``True`` and ``predict_proba`` is absent.
    """
    if not callable(getattr(model, "predict", None)):
        raise TypeError(
            f"model of type '{type(model).__name__}' must implement a callable "
            "``predict(X)`` method (sklearn estimator interface required)."
        )

    if require_predict_proba and not callable(getattr(model, "predict_proba", None)):
        raise AttributeError(
            f"model of type '{type(model).__name__}' does not implement "
            "``predict_proba``.  Use a probabilistic classifier (e.g., "
            "RandomForestClassifier, SVC(probability=True)) or disable "
            "confidence surface rendering."
        )


def validate_class_names(
    class_names: Optional[Dict],
    unique_classes: np.ndarray,
) -> Optional[Dict]:
    """Validate an optional class-name mapping.

    Parameters
    ----------
    class_names : dict or None
        Mapping from class label to display string.
    unique_classes : np.ndarray
        Array of unique class labels detected in the data.

    Returns
    -------
    class_names : dict or None
        The input unchanged if valid.

    Raises
    ------
    TypeError
        If *class_names* is not a dict.
    """
    if class_names is None:
        return None
    if not isinstance(class_names, dict):
        raise TypeError(
            f"class_names must be a dict mapping class labels to strings; "
            f"got {type(class_names).__name__}."
        )
    return class_names

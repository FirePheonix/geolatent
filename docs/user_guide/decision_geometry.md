# Decision Geometry

## The problem with 2-D slices

Most visualisation libraries plot decision boundaries by fixing all features
except two, sweeping a 2-D grid, and colouring cells by predicted class.
This produces a clean picture but shows a single axis-aligned cross-section
of the full decision manifold — it tells you nothing about what happens in
the other n−2 dimensions.

## How GeoLatent does it

GeoLatent constructs the actual decision manifold in the principal-component
subspace of the data:

1. **Project** — fit a {class}`~geolatent.core.projector.DimensionalityProjector`
   on `X` to obtain a 3-D coordinate representation.
2. **Mesh** — build a regular 3-D grid in the projected space (default 30³ = 27 000 points).
3. **Inverse-transform** — map every grid point back to the original n-dimensional
   feature space using `PCA.inverse_transform` or the sensitivity linear map.
4. **Query** — evaluate `model.predict_proba` on all reconstructed feature vectors.
5. **Render** — draw per-class isosurfaces at P = 0.50 (decision boundary),
   P = 0.70 and P = 0.85 (confidence shells).

The result is a genuine decision boundary in the subspace spanned by the three
most informative directions — not a slice.

## Projection methods

`projection_method` controls how the 3-D subspace is chosen.

### `"pca"` (default)

Standard PCA on `X`.  Axes capture the directions of maximum **data variance**.
Good baseline; axes may not align with the model's decision function.

### `"sensitivity"` (recommended)

Finite-difference Jacobians of the model are used to find the directions of
maximum **decision sensitivity**.  See {doc}`sensitivity_projection` for details.

### `"tsne"` / `"umap"`

Non-linear projections for scatter visualisation only.  They do not support
`inverse_transform`, so decision-surface rendering is disabled automatically;
GeoLatent renders a scatter + ellipsoid scene instead.

## Confidence shells

When `show_confidence=True`, three nested isosurface layers are rendered:

| Opacity | Probability threshold | Meaning |
|---|---|---|
| 0.15 | P = 0.50 | Decision boundary |
| 0.25 | P = 0.70 | High-confidence region |
| 0.35 | P = 0.85 | Very-high-confidence region |

## Structural overlays

| Parameter | Description |
|---|---|
| `show_scatter` | Raw data points coloured by class |
| `show_centroids` | Class centroid markers |
| `show_ellipsoids` | Mahalanobis ellipsoids at `ellipsoid_confidence` |
| `show_convex_hulls` | Convex hull polyhedra per class |

## Example

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from geolatent import visualize_decision_geometry

cancer = load_breast_cancer()
gbm = GradientBoostingClassifier(n_estimators=150, max_depth=3).fit(
    cancer.data, cancer.target
)

fig = visualize_decision_geometry(
    gbm, cancer.data, cancer.target,
    projection_method="sensitivity",
    feature_names=list(cancer.feature_names),
    class_names={0: "Malignant", 1: "Benign"},
    show_confidence=True,
    show_centroids=True,
    show_ellipsoids=True,
    mesh_resolution=30,
)
fig.show()
```

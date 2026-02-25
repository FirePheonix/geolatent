# Latent Space Analysis

{func}`~geolatent.api.latent.inspect_latent_space` is designed for exploring the
geometric structure of high-dimensional representation spaces: transformer hidden
states, VAE latent codes, contrastive learning embeddings, GAN feature maps, etc.

## What it renders

- **Scatter cloud** — one point per sample, coloured by class label.
- **Class centroids** — diamond markers at the weighted centre of each cluster.
- **Mahalanobis ellipsoids** — confidence ellipsoids that account for the full
  covariance structure of each class, not just its spread along each axis.
- **Convex hulls** — transparent polyhedra bounding each class cluster.
- **Variance annotation** — the explained variance ratios for the three projected
  dimensions (shown for PCA only).

## Projection methods

All three methods are supported: `"pca"`, `"tsne"`, `"umap"`.

For embeddings with a known low-dimensional signal structure (e.g. cluster means
separated along a few axes), set `scale_input=False` to prevent
{class}`~sklearn.preprocessing.StandardScaler` from equalising variance across
all dimensions — which would wash out the signal.

```python
cfg = DARK_SCIENTIFIC.copy()
cfg.projection.scale_input = False

fig = inspect_latent_space(embeddings, labels, config=cfg.with_method("pca"))
```

## Example — sentence embeddings

```python
import numpy as np
from geolatent import inspect_latent_space, DARK_SCIENTIFIC

rng = np.random.default_rng(42)
D = 64

cluster_means = [
    np.array([0, 0, 0]   + [0] * (D - 3), dtype=float),
    np.array([8, 0, 0]   + [0] * (D - 3), dtype=float),
    np.array([0, 8, 0]   + [0] * (D - 3), dtype=float),
    np.array([4, 4, 6]   + [0] * (D - 3), dtype=float),
]
noise = np.full(D, 0.15)
noise[:3] = 0.7

embeddings = np.vstack([
    rng.normal(size=(200, D)) * noise + m for m in cluster_means
])
labels = np.repeat([0, 1, 2, 3], 200)

cfg = DARK_SCIENTIFIC.copy()
cfg.projection.scale_input = False

for method in ("pca", "tsne", "umap"):
    inspect_latent_space(
        embeddings, labels,
        config=cfg.with_method(method),
        show_ellipsoids=True,
        show_convex_hulls=(method == "pca"),
        class_names={0: "Science", 1: "Politics", 2: "Arts", 3: "Sport"},
        title=f"Sentence embeddings — {method.upper()}",
    ).show()
```

## Ellipsoid confidence

The `ellipsoid_confidence` parameter (default 0.90) sets the probability mass
enclosed by each Mahalanobis ellipsoid, assuming a Gaussian class distribution.
Lower values produce smaller ellipsoids that highlight the dense core of each
cluster.

```python
fig = inspect_latent_space(
    embeddings, labels,
    show_ellipsoids=True,
    ellipsoid_confidence=0.70,   # tighter ellipsoids
)
```

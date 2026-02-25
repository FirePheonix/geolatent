# Quick Start

## Decision geometry of a classifier

```python
from sklearn.svm import SVC
from sklearn.datasets import load_wine
from geolatent import visualize_decision_geometry

wine = load_wine()
model = SVC(kernel="rbf", C=10, gamma="scale", probability=True).fit(
    wine.data, wine.target
)

fig = visualize_decision_geometry(
    model, wine.data, wine.target,
    projection_method="sensitivity",
    feature_names=list(wine.feature_names),
    class_names=dict(enumerate(wine.target_names)),
    show_confidence=True,
    show_ellipsoids=True,
)
fig.show()
```

The `sensitivity` projection finds the three directions in the 13-dimensional
wine feature space where the SVM's decision function changes fastest.
Decision-boundary isosurfaces are drawn at the P = 0.50 boundary, with
optional confidence shells at P = 0.70 and P = 0.85.

## Latent space of embeddings

```python
import numpy as np
from geolatent import inspect_latent_space

# 512 samples of 768-D embeddings (e.g. sentence transformers)
rng = np.random.default_rng(0)
embeddings = np.vstack([
    rng.normal(loc=i * 8, scale=1.0, size=(128, 768))
    for i in range(4)
])
labels = np.repeat([0, 1, 2, 3], 128)

fig = inspect_latent_space(
    embeddings, labels,
    projection_method="umap",
    class_names={0: "Science", 1: "Politics", 2: "Arts", 3: "Sport"},
    show_ellipsoids=True,
    show_convex_hulls=True,
)
fig.show()
```

## Custom theme

All visual properties are controlled via
{class}`~geolatent.config.themes.VisualizationConfig`:

```python
from geolatent import DARK_SCIENTIFIC

cfg = (
    DARK_SCIENTIFIC
    .with_method("tsne")
    .with_title("My Embeddings")
    .with_resolution(1400, 900)
    .with_opacity(surface=0.35, scatter=0.95)
)

fig = inspect_latent_space(embeddings, labels, config=cfg)
```

## What to read next

- {doc}`user_guide/sensitivity_projection` — how model-driven axes work
- {doc}`user_guide/decision_geometry` — the full decision-surface pipeline
- {doc}`tutorials/index` — runnable Colab notebooks

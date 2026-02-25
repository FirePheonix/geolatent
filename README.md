<div align="center">

# GeoLatent

**Geometry-aware, model-intelligent 3-D visualisations for machine learning workflows.**

[![PyPI](https://img.shields.io/pypi/v/geolatent?color=blue&label=PyPI)](https://pypi.org/project/geolatent/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Plotly](https://img.shields.io/badge/rendering-Plotly-3F4F75.svg)](https://plotly.com)
[![Docs](https://img.shields.io/readthedocs/geolatent?label=docs)](https://geolatent.readthedocs.io)
[![GitHub stars](https://img.shields.io/github/stars/FirePheonix/geolatent?style=social)](https://github.com/FirePheonix/geolatent)

</div>

---

## Overview

Most visualisation libraries for machine learning treat the problem superficially:
scatter a few 2-D projections, maybe a confusion matrix, call it done.
**GeoLatent** takes a fundamentally different approach.

It operates as a *semantic abstraction layer* that understands the intrinsic
geometry of models and embeddings.  Rather than plotting model outputs directly,
it constructs the analytical structures that govern model behaviour — decision
manifolds, probability isosurfaces, Mahalanobis confidence regions — and renders
them as first-class geometric objects in interactive 3-D scenes.

### What makes it different

| Capability | Typical wrappers | geolatent |
|---|---|---|
| Projection method | Fixed 2-D PCA | PCA · t-SNE · UMAP, auto-selected |
| Decision surfaces | Axis-aligned slices | True 3-D isosurfaces via PCA inverse-transform |
| Confidence regions | None | Nested probability shells + Mahalanobis ellipsoids |
| Model interface | Specific models only | Any `predict` / `predict_proba` estimator |
| Theme | Matplotlib defaults | Dark-scientific, research-publication quality |
| Notebook support | Requires extra setup | Native Colab/Jupyter inline rendering |

---

## Installation

```bash
pip install geolatent
```

For UMAP support:
```bash
pip install "geolatent[umap]"
```

---

## Quick Start

### Decision geometry of a kernel SVM

```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from geolatent import visualize_decision_geometry

X, y = make_classification(
    n_samples=400, n_features=20, n_classes=3,
    n_informative=10, random_state=42
)
model = SVC(kernel="rbf", C=5.0, probability=True).fit(X, y)

fig = visualize_decision_geometry(
    model, X, y,
    title="RBF-SVM — 3-class Decision Geometry",
    show_confidence=True,
    show_ellipsoids=True,
)
fig.show()
```

### Latent-space geometry of high-dimensional embeddings

```python
import numpy as np
from geolatent import inspect_latent_space

# 512 samples of 768-dimensional embeddings (e.g., BERT sentence vectors)
rng = np.random.default_rng(0)
embeddings = np.vstack([
    rng.normal(loc=mu, scale=1.2, size=(128, 768))
    for mu in [0, 4, 8, 12]
])
labels = np.repeat([0, 1, 2, 3], 128)

fig = inspect_latent_space(
    embeddings, labels,
    projection_method="pca",
    title="768-D Embeddings — 4 Topic Clusters",
    class_names={0: "Science", 1: "Politics", 2: "Arts", 3: "Sports"},
)
fig.show()
```

---

## Architecture

geolatent separates three orthogonal concerns into dedicated sub-packages:

```
geolatent/
├── config/
│   └── themes.py          ← ColorPalette · RenderConfig · ProjectionConfig
│                            VisualizationConfig · DARK_SCIENTIFIC
├── core/
│   ├── projector.py       ← DimensionalityProjector (PCA / t-SNE / UMAP)
│   ├── mesh_builder.py    ← MeshBuilder · PredictionMesh
│   └── geometry.py        ← GeometryUtils (ellipsoids · centroids · hulls)
├── rendering/
│   ├── scene.py           ← Scene3D (Plotly figure manager)
│   ├── surfaces.py        ← DecisionSurfaceRenderer (isosurfaces · volumes)
│   └── overlays.py        ← DataOverlay (scatter · centroids · ellipsoids)
└── api/
    ├── decision.py        ← visualize_decision_geometry()
    └── latent.py          ← inspect_latent_space()
```

### The decision-surface pipeline

The key insight that separates geolatent from shallow wrappers is the
**inverse-transform prediction mesh**.  For a model trained on `n_features`-
dimensional data:

1. Fit PCA on `X` → extract top 3 principal components.
2. Construct a regular 3-D grid in the PC space (e.g., 30³ = 27 000 points).
3. Apply `PCA.inverse_transform` → map each grid point back to the original
   `n_features`-dimensional feature space.
4. Query `model.predict_proba` on the reconstructed feature vectors.
5. Render per-class probability isosurfaces at the P = 0.50 decision boundary,
   plus optional confidence shells at P = 0.70 and P = 0.85.

This produces genuine decision boundaries that reflect the model's actual
behaviour in the subspace spanned by the top 3 principal directions — not an
approximation based on a 2-D axis-aligned slice.

---

## Configuration

All styling and algorithmic parameters are controlled through `VisualizationConfig`,
a nested dataclass that can be customised via fluent helpers:

```python
from geolatent import VisualizationConfig, DARK_SCIENTIFIC

cfg = (
    DARK_SCIENTIFIC
    .with_method("tsne")                  # t-SNE projection
    .with_title("GBM Latent Geometry")    # figure title
    .with_resolution(1200, 800)           # canvas size
    .with_opacity(surface=0.4, scatter=0.9)
)

fig = inspect_latent_space(embeddings, labels, config=cfg)
```

### Colour palette

The default `DARK_SCIENTIFIC` theme uses a GitHub Dark–inspired palette with
eight class colours chosen for accessibility and contrast on dark backgrounds:

```
#58a6ff  #3fb950  #f78166  #d2a8ff
#ffa657  #79c0ff  #56d364  #ff7b72
```

---

## Advanced usage

### Custom pipeline (lower-level API)

For research workflows that need fine-grained control:

```python
from geolatent.core.projector import DimensionalityProjector
from geolatent.core.mesh_builder import MeshBuilder
from geolatent.rendering.scene import Scene3D
from geolatent.rendering.surfaces import DecisionSurfaceRenderer
from geolatent.rendering.overlays import DataOverlay
from geolatent import DARK_SCIENTIFIC

cfg = DARK_SCIENTIFIC.copy()
cfg.projection.method = "pca"

# 1. Project
projector = DimensionalityProjector(cfg.projection)
result = projector.fit_transform(X)

# 2. Build mesh
mesh = MeshBuilder(resolution=35).build_prediction_mesh(clf, projector, result.coordinates)

# 3. Compose scene
scene = Scene3D(cfg)
scene.set_axis_labels(result.axis_labels)
scene.add_traces(DecisionSurfaceRenderer(cfg).render(mesh))
scene.add_traces(DataOverlay(cfg).render_scatter(result.coordinates, y))
scene.add_trace(DataOverlay(cfg).render_centroids(result.coordinates, y))
scene.add_variance_annotation(result.explained_variance_ratio)
fig = scene.render()
fig.show()
```

### Optimisation trajectories

```python
from geolatent.rendering.overlays import DataOverlay

overlay = DataOverlay(cfg)
# waypoints: projected coordinates of gradient-descent iterates
trajectory_traces = overlay.render_trajectory(waypoints, name="SGD path")
scene.add_traces(trajectory_traces)
```

---

## Design principles

**Semantic over syntactic.**  The API speaks in ML concepts — models, embeddings,
decision boundaries, confidence regions — not in Plotly trace types.

**Modularity over monolithism.**  Projection, mesh construction, geometry, and
rendering are four independent modules with clearly defined interfaces.  Each
can be replaced, extended, or tested in isolation.

**Correctness over speed.**  The decision-surface computation is geometrically
exact within the PCA subspace; we do not approximate boundaries by sampling
random projections or evaluating only on 2-D slices.

**Research-grade aesthetics.**  The dark-scientific theme is designed to produce
figures suitable for ML conference supplementary materials without any
post-processing.

---

## Requirements

| Dependency | Version |
|---|---|
| Python | ≥ 3.9 |
| NumPy | ≥ 1.23 |
| SciPy | ≥ 1.9 |
| scikit-learn | ≥ 1.1 |
| Plotly | ≥ 5.13 |
| umap-learn *(optional)* | ≥ 0.5.3 |

---

## Citation

If geolatent contributes to published research, please acknowledge it:

```bibtex
@software{geolatent2026,
  title  = {GeoLatent: Geometry-aware 3-D Visualisations for Machine Learning},
  year   = {2026},
  url    = {https://pypi.org/project/geolatent/},
}
```

---

## License

MIT — see [LICENSE](LICENSE).

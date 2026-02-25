# GeoLatent

**Geometry-aware, model-intelligent 3-D visualisations for machine learning workflows.**

[![PyPI](https://img.shields.io/pypi/v/geolatent.svg)](https://pypi.org/project/geolatent/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/FirePheonix/geolatent/blob/main/LICENSE)

---

GeoLatent renders the true 3-D decision boundary of any classifier and the geometric
structure of any embedding space — in a single function call.

```python
from geolatent import visualize_decision_geometry
from sklearn.svm import SVC

model = SVC(kernel="rbf", probability=True).fit(X, y)
fig = visualize_decision_geometry(model, X, y, projection_method="sensitivity")
fig.show()
```

## Why GeoLatent?

| | Standard wrappers | GeoLatent |
|---|---|---|
| Projection | Fixed 2-D PCA | PCA · t-SNE · UMAP · Sensitivity |
| Decision surfaces | Axis-aligned slices | True 3-D isosurfaces via inverse-transform |
| Confidence regions | None | Nested probability shells + Mahalanobis ellipsoids |
| Model interface | sklearn only | Any `predict` / `predict_proba` callable |
| Projection axes | Data-driven | **Model-driven** via finite-difference Jacobians |

The sensitivity projection is GeoLatent's key differentiator: it computes
finite-difference Jacobians of any model to find the three directions in feature
space where the decision function changes fastest — axes that are task-relevant
regardless of input dimensionality.

---

## Install

```bash
pip install geolatent
```

With UMAP support:

```bash
pip install "geolatent[umap]"
```

---

```{toctree}
:maxdepth: 1
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guide

user_guide/decision_geometry
user_guide/latent_space
user_guide/sensitivity_projection
user_guide/configuration
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/index
```

```{toctree}
:maxdepth: 1
:caption: Reference

autoapi/index
changelog
```

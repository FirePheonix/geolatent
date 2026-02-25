# Changelog

## 0.1.0 (2026-02-25)

Initial release.

### Added

- `visualize_decision_geometry()` — 3-D decision boundary isosurfaces via PCA / sensitivity inverse-transform
- `inspect_latent_space()` — latent space visualisation with PCA / t-SNE / UMAP
- Sensitivity projection — model-driven axes via finite-difference Jacobians, works with any callable
- `feature_names` support — axis labels show top contributing features
- Mahalanobis confidence ellipsoids at configurable probability levels
- Convex hull overlays per class
- Optimisation trajectory rendering
- `DARK_SCIENTIFIC` theme — dark-scientific colour palette with fluent config helpers
- Full type annotations and `py.typed` marker

# Configuration

All visual and algorithmic parameters live in
{class}`~geolatent.config.themes.VisualizationConfig`, a nested dataclass.
The default theme is `DARK_SCIENTIFIC`.

## Structure

```
VisualizationConfig
├── colors      : ColorPalette     — class colours, background, grid
├── render      : RenderConfig     — canvas size, opacity, marker size
└── projection  : ProjectionConfig — method, scale_input, t-SNE / UMAP params
```

## Fluent helpers

The `DARK_SCIENTIFIC` singleton provides helpers that return a modified copy:

```python
from geolatent import DARK_SCIENTIFIC

cfg = (
    DARK_SCIENTIFIC
    .with_method("tsne")
    .with_title("My Figure")
    .with_resolution(1400, 900)
    .with_opacity(surface=0.3, scatter=0.9)
)
```

## Manual configuration

For full control, construct or copy the config and mutate its fields:

```python
cfg = DARK_SCIENTIFIC.copy()

# Projection
cfg.projection.method             = "sensitivity"
cfg.projection.scale_input        = True
cfg.projection.sensitivity_n_subsample = 500
cfg.projection.tsne_perplexity    = 40.0
cfg.projection.tsne_n_iter        = 1000

# Rendering
cfg.render.width        = 1200
cfg.render.height       = 800
cfg.render.surface_opacity = 0.25
cfg.render.scatter_opacity = 0.95
cfg.render.marker_size  = 4
```

## Custom colour palette

```python
from geolatent.config.themes import ColorPalette, VisualizationConfig, RenderConfig, ProjectionConfig

my_colors = ColorPalette(
    background="#0a0a0a",
    grid="#1a1a2e",
    text="#ffffff",
    axis_line="#333366",
    class_colors=["#e63946", "#457b9d", "#2a9d8f", "#e9c46a"],
    annotation_color="#8888aa",
)

cfg = VisualizationConfig(
    colors=my_colors,
    render=RenderConfig(width=1100, height=750),
    projection=ProjectionConfig(method="pca"),
)
```

## ProjectionConfig reference

| Field | Type | Default | Description |
|---|---|---|---|
| `method` | str | `"pca"` | `"pca"` · `"tsne"` · `"umap"` · `"sensitivity"` |
| `scale_input` | bool | `True` | Standardise features before projection |
| `n_components` | int | `3` | Number of output dimensions |
| `tsne_perplexity` | float | `30.0` | t-SNE perplexity |
| `tsne_n_iter` | int | `1000` | t-SNE iterations |
| `umap_n_neighbors` | int | `15` | UMAP neighbourhood size |
| `umap_min_dist` | float | `0.1` | UMAP minimum distance |
| `sensitivity_n_subsample` | int | `300` | Points used for Jacobian estimation |
| `sensitivity_eps` | float | `0.01` | Finite-difference perturbation magnitude |

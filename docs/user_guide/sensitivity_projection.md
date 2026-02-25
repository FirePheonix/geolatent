# Sensitivity Projection

Sensitivity projection is GeoLatent's model-driven dimensionality reduction.
Unlike PCA — which finds directions of maximum data variance — sensitivity
projection finds the directions where the **model's output changes fastest**.

## Motivation

Consider a 30-feature breast cancer classifier.  PCA on the 30 features will
surface the axes that explain the most variance in the data.  These may be
dominated by features that happen to vary a lot but are not particularly
predictive.  Sensitivity projection asks a different question: *which
directions in the 30-D space, if you perturbed a patient along them, would
most change the model's malignancy probability?*  Those axes are directly
interpretable in terms of the model's decision function.

## Algorithm

Given a model `f : ℝⁿ → ℝ` (or `ℝⁿ → ℝᶜ` for multi-class) and a dataset `X`:

1. **Subsample** — draw `sensitivity_n_subsample` points uniformly from `X`
   (default 300).
2. **Finite differences** — for each subsample point `xᵢ` and each feature `j`,
   perturb by `ε` and measure the change in model output:

   ```
   S[i, j] = ‖f(xᵢ + ε·eⱼ) − f(xᵢ)‖ / ε
   ```

3. **Sensitivity matrix** — `S` has shape `(n_subsample, n_features)`.  Each
   row is a sensitivity profile for one data point.
4. **PCA on S** — extract the top 3 principal components of `S`.  These are
   linear combinations of features with maximum sensitivity variance across
   the population.
5. **Project** — apply the linear map `(X − μ) @ components.T` where `μ` is
   the feature-space mean.

The resulting projection is **linear** and **invertible**, so decision-surface
meshes can be inverse-transformed back to the original feature space exactly
like PCA.

## Properties

- Works for **any callable** model: sklearn, PyTorch, XGBoost, custom lambdas.
- **Invertible** — decision-surface rendering is fully supported.
- **Named axes** — axis labels show the top contributing features, e.g.
  `"Sens 1 (worst_radius, mean_concavity)"`.
- Scales to any input dimensionality.

## Usage

```python
fig = visualize_decision_geometry(
    model, X, y,
    projection_method="sensitivity",
    feature_names=feature_names,   # optional but recommended for axis labels
)
```

For non-sklearn models, pass a custom `predict_fn`:

```python
import torch

def torch_predict(X_np):
    with torch.no_grad():
        logits = my_pytorch_model(torch.tensor(X_np, dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()

fig = visualize_decision_geometry(
    model=None,            # not used when predict_fn is provided
    X=X, y=y,
    projection_method="sensitivity",
    predict_fn=torch_predict,
    feature_names=feature_names,
)
```

## Tuning

| Parameter | Default | Effect |
|---|---|---|
| `sensitivity_n_subsample` | 300 | More points → more stable Jacobians, slower |
| `sensitivity_eps` | 0.01 | Perturbation magnitude; smaller = more local |

Set these via {class}`~geolatent.config.themes.ProjectionConfig`:

```python
from geolatent import DARK_SCIENTIFIC

cfg = DARK_SCIENTIFIC.copy()
cfg.projection.sensitivity_n_subsample = 500
cfg.projection.sensitivity_eps = 0.005

fig = visualize_decision_geometry(model, X, y, config=cfg,
                                  projection_method="sensitivity")
```

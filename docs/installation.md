# Installation

## Requirements

| Package | Version |
|---|---|
| Python | ≥ 3.9 |
| NumPy | ≥ 1.23 |
| SciPy | ≥ 1.9 |
| scikit-learn | ≥ 1.1 |
| Plotly | ≥ 5.13 |

## pip

```bash
pip install geolatent
```

## Optional dependencies

UMAP projection requires `umap-learn`:

```bash
pip install "geolatent[umap]"
```

Static image export (PNG, SVG) requires `kaleido`:

```bash
pip install "geolatent[export]"
```

Install everything at once:

```bash
pip install "geolatent[umap,export]"
```

## Google Colab

```python
!pip install -q geolatent
# Restart runtime after installation, then import normally
```

## Verify

```python
import geolatent
print(geolatent.__version__)
```

## Development install

```bash
git clone https://github.com/FirePheonix/geolatent.git
cd geolatent
pip install -e ".[dev]"
```

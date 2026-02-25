# Contributing to GeoLatent

Thank you for your interest in contributing.  This guide covers everything
needed to submit bug fixes, new features, and documentation improvements.

---

## Table of contents

- [Development setup](#development-setup)
- [Running tests](#running-tests)
- [Code style](#code-style)
- [Submitting a pull request](#submitting-a-pull-request)
- [Contributing to the docs](#contributing-to-the-docs)
- [Adding a tutorial notebook](#adding-a-tutorial-notebook)
- [Reporting a bug](#reporting-a-bug)

---

## Development setup

```bash
git clone https://github.com/FirePheonix/geolatent.git
cd geolatent
pip install -e ".[umap,dev]"
```

This installs GeoLatent in editable mode along with all development tools
(pytest, black, ruff, mypy).

---

## Running tests

```bash
pytest
```

Coverage report:

```bash
pytest --cov=geolatent --cov-report=term-missing
```

---

## Code style

GeoLatent uses [Black](https://black.readthedocs.io) for formatting and
[Ruff](https://docs.astral.sh/ruff/) for linting.

```bash
black geolatent/
ruff check geolatent/
```

Type annotations are checked with mypy:

```bash
mypy geolatent/
```

All three must pass before a pull request will be merged.

---

## Submitting a pull request

1. Fork the repository and create a branch from `main`:
   ```bash
   git checkout -b fix/my-bug-fix
   ```
2. Make your changes.  For bug fixes, add a test that reproduces the issue.
3. Run the full check suite (`pytest`, `black`, `ruff`, `mypy`).
4. Push and open a pull request against `main`.

**Pull request checklist**

- [ ] Tests pass locally
- [ ] Code formatted with Black, no Ruff warnings
- [ ] New public functions/classes have NumPy-style docstrings
- [ ] `docs/changelog.md` updated with a one-line summary under the next version

---

## Contributing to the docs

The documentation lives in `docs/` and is built with Sphinx.

### Install docs dependencies

```bash
pip install -e ".[docs]"
```

### Build locally

```bash
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser to preview.

### Check for broken links

```bash
sphinx-build -b linkcheck docs docs/_build/linkcheck
```

### Doc structure

| Path | Purpose |
|---|---|
| `docs/index.md` | Landing page and top-level toctree |
| `docs/installation.md` | Install instructions |
| `docs/quickstart.md` | Two-minute introduction |
| `docs/user_guide/` | Conceptual explanations |
| `docs/tutorials/index.md` | Index of tutorial notebooks |
| `docs/changelog.md` | Version history |
| `docs/conf.py` | Sphinx configuration |

API reference pages are **auto-generated** from docstrings by `sphinx-autoapi`
— do not edit anything under `docs/autoapi/`.

### Docstring format

Use NumPy-style docstrings:

```python
def my_function(X: np.ndarray, n: int = 3) -> np.ndarray:
    """One-line summary.

    Longer description if needed.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input feature matrix.
    n : int, optional
        Number of components.  Default is 3.

    Returns
    -------
    np.ndarray of shape (n_samples, n)
        Transformed coordinates.

    Examples
    --------
    >>> result = my_function(X)
    """
```

---

## Adding a tutorial notebook

1. Create a new notebook under `examples/`:
   ```
   examples/my_topic.ipynb
   ```
2. Follow the style of existing notebooks:
   - First cell: `!pip install -q geolatent` (or `geolatent[umap]`)
   - No numbered cell comments (`# Cell 1 —` etc.)
   - Markdown cells explain the *why*, not the *what*
   - No local file paths or environment-specific references
3. Verify it runs end-to-end on a fresh Colab runtime.
4. Add an entry to `docs/tutorials/index.md` with the Colab badge:
   ```markdown
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FirePheonix/geolatent/blob/main/examples/my_topic.ipynb)
   ```
5. Commit both the notebook and the updated tutorials index.

---

## Reporting a bug

Open an issue at <https://github.com/FirePheonix/geolatent/issues> and include:

- GeoLatent version (`import geolatent; print(geolatent.__version__)`)
- Python version and OS
- Minimal reproducible example
- Full traceback

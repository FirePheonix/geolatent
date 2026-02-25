# Tutorials

Each tutorial is a runnable Colab notebook.  Click the badge to open it directly
in Google Colab — no local setup required.

---

## Decision Geometry

**[decision_geometry.ipynb](https://github.com/FirePheonix/geolatent/blob/main/examples/decision_geometry.ipynb)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FirePheonix/geolatent/blob/main/examples/decision_geometry.ipynb)

Real-world datasets (Wine, Breast Cancer, Digits, Iris) with multiple model
families.  Compares PCA and sensitivity projection side by side.

---

## Latent Space Explorer

**[latent_space.ipynb](https://github.com/FirePheonix/geolatent/blob/main/examples/latent_space.ipynb)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FirePheonix/geolatent/blob/main/examples/latent_space.ipynb)

Document embeddings, protein family vectors, and market regime states — all
in 64–128 dimensions.  PCA, t-SNE, and UMAP projections compared.

---

## Synthetic Decision Boundaries

**[synthetic_boundaries.ipynb](https://github.com/FirePheonix/geolatent/blob/main/examples/synthetic_boundaries.ipynb)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FirePheonix/geolatent/blob/main/examples/synthetic_boundaries.ipynb)

Two moons, concentric circles, anisotropic blobs, and XOR — each embedded in
20-D noise.  Demonstrates sensitivity projection recovering the informative
axes from pure noise.

---

## High-Dimensional Feature Spaces

**[high_dimensional.ipynb](https://github.com/FirePheonix/geolatent/blob/main/examples/high_dimensional.ipynb)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FirePheonix/geolatent/blob/main/examples/high_dimensional.ipynb)

Olivetti faces (4 096-D), 20 Newsgroups TF-IDF (300-D), and MNIST digits
(784-D).  Shows sensitivity projection on real high-dimensional data.

---

## Model Comparison

**[model_comparison.ipynb](https://github.com/FirePheonix/geolatent/blob/main/examples/model_comparison.ipynb)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FirePheonix/geolatent/blob/main/examples/model_comparison.ipynb)

Six model families — Logistic Regression, Linear SVM, RBF SVM, Random Forest,
Gradient Boosting, MLP — trained on the same dataset and visualised side by side
in both sensitivity and PCA frames.

---

## Geographically Weighted PCA — Synthetic

**[gwpca.ipynb](https://github.com/FirePheonix/geolatent/blob/main/examples/gwpca.ipynb)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FirePheonix/geolatent/blob/main/examples/gwpca.ipynb)

GWPCA implemented from scratch on a spatially structured synthetic dataset.
Bandwidth sensitivity analysis and comparison with global PCA.

---

## Geographically Weighted PCA — NYC Airbnb

**[gwpca_nyc_airbnb.ipynb](https://github.com/FirePheonix/geolatent/blob/main/examples/gwpca_nyc_airbnb.ipynb)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FirePheonix/geolatent/blob/main/examples/gwpca_nyc_airbnb.ipynb)

GWPCA on the real Kaggle NYC Airbnb dataset (48 895 listings).  Local PC
explained variance mapped across the five boroughs; loading interpretation
reveals borough-specific market dynamics.

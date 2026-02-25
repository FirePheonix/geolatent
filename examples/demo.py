"""
GeoLatent demo
Run: python examples/demo.py
"""

import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer, load_digits
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import geolatent as nv
from geolatent import visualize_decision_geometry, inspect_latent_space, DARK_SCIENTIFIC

print(f"GeoLatent v{nv.__version__} ready.\n")

wine   = load_wine()
cancer = load_breast_cancer()
digits = load_digits()


# Demo 1: Wine SVM, PCA (data-driven axes)
print("[1/5] Wine SVM — PCA ...")
svm_wine = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=0)
svm_wine.fit(wine.data, wine.target)

visualize_decision_geometry(
    model=svm_wine,
    X=wine.data,
    y=wine.target,
    projection_method="pca",
    feature_names=list(wine.feature_names),
    class_names=dict(enumerate(wine.target_names)),
    show_confidence=True,
    show_centroids=True,
    show_ellipsoids=True,
    title="Wine — SVM Decision Boundary (PCA: data-driven axes)",
).show()


# Demo 2: Wine SVM, Sensitivity (model-driven axes)
print("[2/5] Wine SVM — Sensitivity ...")
print("      computing Jacobians (~5 sec)")

visualize_decision_geometry(
    model=svm_wine,
    X=wine.data,
    y=wine.target,
    projection_method="sensitivity",
    feature_names=list(wine.feature_names),
    class_names=dict(enumerate(wine.target_names)),
    show_confidence=True,
    show_centroids=True,
    show_ellipsoids=True,
    title="Wine — SVM Decision Boundary (Sensitivity: model-driven axes)",
).show()


# Demo 3: Breast cancer GBM, Sensitivity (30 named features)
print("[3/5] Breast cancer GBM — Sensitivity (30-D) ...")

gbm_cancer = GradientBoostingClassifier(
    n_estimators=100, max_depth=3, random_state=0
).fit(cancer.data, cancer.target)

visualize_decision_geometry(
    model=gbm_cancer,
    X=cancer.data,
    y=cancer.target,
    projection_method="sensitivity",
    feature_names=list(cancer.feature_names),
    class_names={0: "Malignant", 1: "Benign"},
    show_confidence=True,
    show_centroids=True,
    show_ellipsoids=True,
    title="Breast Cancer — GBM Decision Boundary (Sensitivity)",
).show()


# Demo 4: Handwritten digits, RF, Sensitivity (64 pixel features)
print("[4/5] Digits RF — Sensitivity (64 pixels, first 4 classes) ...")

mask  = digits.target < 4
X_dig = digits.data[mask]
y_dig = digits.target[mask]

rf_digits = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_dig, y_dig)
pixel_names = [f"px_{i//8}_{i%8}" for i in range(64)]

visualize_decision_geometry(
    model=rf_digits,
    X=X_dig,
    y=y_dig,
    projection_method="sensitivity",
    feature_names=pixel_names,
    class_names={0: "Digit 0", 1: "Digit 1", 2: "Digit 2", 3: "Digit 3"},
    show_confidence=False,
    show_centroids=True,
    show_ellipsoids=True,
    title="Handwritten Digits (0-3) — RF Sensitivity on 64 pixels",
).show()


# Demo 5: 64-D embeddings, PCA / t-SNE / UMAP
# Signal in dims 0-2, noise in dims 3-63 (std=0.15).
# scale_input=False so PCA finds the signal dims directly.
print("[5/5] 64-D embeddings — PCA / t-SNE / UMAP ...")

rng = np.random.default_rng(0)
D = 64
cluster_means = [
    np.zeros(D),
    np.array([8.0, 0.0, 0.0] + [0.0] * (D - 3)),
    np.array([0.0, 8.0, 0.0] + [0.0] * (D - 3)),
    np.array([4.0, 4.0, 6.0] + [0.0] * (D - 3)),
]
noise_std = np.full(D, 0.15)
noise_std[:3] = 0.7

embeddings = np.vstack([
    rng.normal(size=(150, D)) * noise_std + m
    for m in cluster_means
])
emb_labels = np.repeat([0, 1, 2, 3], 150)
emb_names  = {0: "Topic: Science", 1: "Topic: Politics",
               2: "Topic: Arts",    3: "Topic: Sport"}

cfg_emb = DARK_SCIENTIFIC.copy()
cfg_emb.projection.scale_input = False

for method in ("pca", "tsne", "umap"):
    print(f"      {method.upper()} ...")
    inspect_latent_space(
        embeddings=embeddings, labels=emb_labels,
        config=cfg_emb.with_method(method),
        show_ellipsoids=True,
        show_convex_hulls=(method == "pca"),
        class_names=emb_names,
        title=f"64-D Embeddings — {method.upper()} Projection",
    ).show()

print("\nDone. 7 figures in browser.")

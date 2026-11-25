"""Cluster simulated customers with a Gaussian Mixture Model."""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


rng = np.random.default_rng(79)
centers = np.array([
    [0.2, 0.8, 0.25, 0.35],
    [0.8, 0.25, 0.7, 0.35],
    [0.45, 0.5, 0.35, 0.85],
])
customer_segments = rng.integers(0, len(centers), size=600)
behaviour = centers[customer_segments] + rng.normal(0, 0.14, size=(600, 4))
features = StandardScaler().fit_transform(behaviour)
model = GaussianMixture(
    n_components=3, covariance_type="full", random_state=79, n_init=5
).fit(features)
probabilities = model.predict_proba(features)
assignments = probabilities.argmax(axis=1)
print("GMM BIC:", round(model.bic(features), 2))
for component in range(model.n_components):
    print(
        f"Component {component + 1}: {np.sum(assignments == component)} customers; "
        f"mean membership probability={probabilities[:, component].mean():.3f}"
    )
print("First five posterior membership vectors:")
print(np.round(probabilities[:5], 3))
print("Each row sums to one and gives soft cluster membership probabilities.")

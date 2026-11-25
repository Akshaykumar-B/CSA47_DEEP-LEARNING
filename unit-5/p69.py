"""Project high-dimensional synthetic clusters into two principal components."""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


features, clusters = make_blobs(
    n_samples=240,
    n_features=10,
    centers=3,
    cluster_std=2.2,
    random_state=69,
)
standardized = StandardScaler().fit_transform(features)
pca = PCA(n_components=2, random_state=69)
coordinates = pca.fit_transform(standardized)
print("Two-dimensional projection (first ten samples; PC1, PC2, generated group):")
for point, label in zip(coordinates[:10], clusters[:10]):
    print(f"({point[0]:+.2f}, {point[1]:+.2f}), group={label}")
print("Explained variance in the displayed axes:", np.round(pca.explained_variance_ratio_, 3))
for label in np.unique(clusters):
    center = coordinates[clusters == label].mean(axis=0)
    print(f"Group {label} projected center: ({center[0]:+.2f}, {center[1]:+.2f})")
print("The printed coordinates provide a text-only view of the first-two-component projection.")

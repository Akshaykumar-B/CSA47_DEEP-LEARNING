"""Apply PCA to correlated synthetic medical attributes and inspect components."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


rng = np.random.default_rng(67)
latent = rng.normal(size=(500, 3))
loadings = rng.normal(size=(3, 12))
attributes = latent @ loadings + rng.normal(0, 0.25, size=(500, 12))
feature_names = [
    "age", "blood_pressure", "glucose", "cholesterol", "bmi", "heart_rate",
    "insulin", "waist", "triglycerides", "oxygen", "activity", "sleep",
]
standardized = StandardScaler().fit_transform(attributes)
pca = PCA(n_components=3, random_state=67)
transformed = pca.fit_transform(standardized)
print("Original feature shape:", attributes.shape)
print("PCA feature shape:", transformed.shape)
print("Explained variance ratios:", np.round(pca.explained_variance_ratio_, 3))
for component_index, component in enumerate(pca.components_, start=1):
    top = np.argsort(np.abs(component))[-3:][::-1]
    print(
        f"PC{component_index} strongest loadings:",
        ", ".join(f"{feature_names[index]} ({component[index]:+.2f})" for index in top),
    )
print("The first components summarize correlated attributes; this dataset is simulated.")

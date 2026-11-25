"""Find how many PCA components retain at least 95 percent of variance."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


rng = np.random.default_rng(71)
latent = rng.normal(size=(800, 5))
mixing = rng.normal(size=(5, 24))
features = latent @ mixing + rng.normal(0, 0.35, size=(800, 24))
standardized = StandardScaler().fit_transform(features)
pca = PCA().fit(standardized)
cumulative = np.cumsum(pca.explained_variance_ratio_)
components_needed = int(np.searchsorted(cumulative, 0.95) + 1)
print(f"Original features: {features.shape[1]}")
print(f"Components needed for at least 95% variance: {components_needed}")
print(f"Variance retained: {cumulative[components_needed - 1]:.3%}")
print("First ten cumulative explained-variance values:", np.round(cumulative[:10], 3))

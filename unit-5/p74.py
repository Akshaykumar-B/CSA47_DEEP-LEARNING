"""Compare PCA and ICA components on a mixed non-Gaussian dataset."""

import numpy as np
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler


rng = np.random.default_rng(74)
count = 2500
sources = np.column_stack([
    rng.uniform(-1, 1, count),
    rng.laplace(size=count),
    np.sign(np.sin(np.linspace(0, 25, count))),
])
sources = StandardScaler().fit_transform(sources)
mixing = np.array([[1.0, 0.6, 0.3], [0.2, 1.0, 0.5], [0.5, 0.1, 1.0]])
observations = sources @ mixing.T
standardized = StandardScaler().fit_transform(observations)
pca = PCA(n_components=3, whiten=True, random_state=74)
pca_components = pca.fit_transform(standardized)
ica = FastICA(n_components=3, random_state=74, max_iter=1500, tol=0.001)
ica_components = ica.fit_transform(standardized)


def mean_off_diagonal_correlation(values):
    correlations = np.corrcoef(values, rowvar=False)
    mask = ~np.eye(correlations.shape[0], dtype=bool)
    return np.abs(correlations[mask]).mean()


print("PCA explained variance ratios:", np.round(pca.explained_variance_ratio_, 3))
print(f"PCA mean absolute component correlation: {mean_off_diagonal_correlation(pca_components):.4f}")
print(f"ICA mean absolute component correlation: {mean_off_diagonal_correlation(ica_components):.4f}")
print("PCA component excess kurtosis:", np.round(kurtosis(pca_components), 2))
print("ICA component excess kurtosis:", np.round(kurtosis(ica_components), 2))
print("PCA captures variance and decorrelates; ICA seeks statistically independent sources.")

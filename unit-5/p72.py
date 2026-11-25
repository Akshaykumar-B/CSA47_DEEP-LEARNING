"""Separate three independent synthetic source signals with FastICA."""

import numpy as np
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA


rng = np.random.default_rng(72)
time = np.linspace(0, 8, 2000)
sources = np.column_stack([
    np.sin(2 * np.pi * 1.0 * time),
    np.sign(np.sin(2 * np.pi * 0.45 * time)),
    rng.laplace(size=len(time)),
])
sources = (sources - sources.mean(axis=0)) / sources.std(axis=0)
mixing = np.array([[1.0, 0.5, 0.2], [0.3, 1.0, 0.6], [0.4, 0.2, 1.0]])
mixed = sources @ mixing.T
ica = FastICA(n_components=3, random_state=72, max_iter=1500, tol=0.001)
separated = ica.fit_transform(mixed)
correlation = np.corrcoef(sources.T, separated.T)[:3, 3:]
for source_index in range(3):
    component_index = int(np.argmax(np.abs(correlation[source_index])))
    print(
        f"Source {source_index + 1} best matches component {component_index + 1}; "
        f"|correlation|={abs(correlation[source_index, component_index]):.3f}, "
        f"separated kurtosis={kurtosis(separated[:, component_index]):.2f}"
    )
print("ICA recovers components up to permutation, sign, and scale.")

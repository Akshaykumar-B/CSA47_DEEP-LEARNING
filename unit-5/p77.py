"""Compare PCA and factor-analysis reconstruction errors on held-out data."""

import numpy as np
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


rng = np.random.default_rng(77)
latent = rng.normal(size=(900, 4))
mixing = rng.normal(size=(4, 14))
features = latent @ mixing + rng.normal(0, 0.5, size=(900, 14))
x_train, x_test = train_test_split(features, test_size=0.25, random_state=77)
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
component_count = 4

pca = PCA(n_components=component_count, random_state=77).fit(x_train)
pca_reconstruction = pca.inverse_transform(pca.transform(x_test))
pca_error = np.mean((x_test - pca_reconstruction) ** 2)

factor_model = FactorAnalysis(n_components=component_count, random_state=77).fit(x_train)
factor_scores = factor_model.transform(x_test)
factor_reconstruction = factor_scores @ factor_model.components_ + factor_model.mean_
factor_error = np.mean((x_test - factor_reconstruction) ** 2)
print(f"Original feature count: {features.shape[1]}; retained dimensions: {component_count}")
print(f"PCA held-out mean squared reconstruction error: {pca_error:.4f}")
print(f"Factor Analysis held-out mean squared reconstruction error: {factor_error:.4f}")
print("Both methods reduce dimensions; PCA targets variance, while Factor Analysis models shared causes.")

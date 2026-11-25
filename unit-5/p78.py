"""Generate multivariate Gaussian sensor data and estimate its parameters."""

import numpy as np


rng = np.random.default_rng(78)
true_mean = np.array([22.0, 101.3, 0.45])
true_covariance = np.array([
    [4.0, 0.8, 0.2],
    [0.8, 2.5, 0.3],
    [0.2, 0.3, 0.09],
])
samples = rng.multivariate_normal(true_mean, true_covariance, size=5000)
estimated_mean = samples.mean(axis=0)
estimated_covariance = np.cov(samples, rowvar=False)
print("True mean:      ", np.round(true_mean, 3))
print("Estimated mean: ", np.round(estimated_mean, 3))
print("True covariance:\n", np.round(true_covariance, 3))
print("Estimated covariance:\n", np.round(estimated_covariance, 3))
print(f"Mean estimation error (Euclidean): {np.linalg.norm(estimated_mean - true_mean):.4f}")
print(f"Covariance estimation error (Frobenius): {np.linalg.norm(estimated_covariance - true_covariance):.4f}")
print("Columns represent temperature, pressure, and vibration sensor measurements.")

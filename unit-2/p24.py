"""Train a single-layer perceptron to classify loan applicants."""

import numpy as np


# Features: annual income in $10,000s and credit score in hundreds.
x = np.array(
    [
        [3.0, 5.0], [4.0, 7.0], [5.0, 5.0], [5.5, 6.0],
        [6.0, 5.5], [6.0, 7.0], [7.0, 6.0], [8.0, 8.0],
    ]
)
y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
mean, scale = x.mean(axis=0), x.std(axis=0)
x_scaled = (x - mean) / scale
weights = np.zeros(x.shape[1])
bias = 0.0

for _ in range(100):
    mistakes = 0
    for features, label in zip(x_scaled, y):
        prediction = int(np.dot(weights, features) + bias >= 0)
        error = label - prediction
        if error:
            weights += error * features
            bias += error
            mistakes += 1
    if mistakes == 0:
        break

for features, actual in zip(x, y):
    prediction = int(np.dot(weights, (features - mean) / scale) + bias >= 0)
    print(
        f"income=${features[0] * 10:.0f}k, credit={features[1] * 100:.0f}: "
        f"predicted={'eligible' if prediction else 'ineligible'}, actual={actual}"
    )
print(f"Training epochs: {_ + 1}; weights={np.round(weights, 3)}, bias={bias:.3f}")

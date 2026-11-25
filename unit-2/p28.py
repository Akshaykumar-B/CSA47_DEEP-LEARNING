"""Compare ReLU, sigmoid, and tanh hidden activations on churn data."""

import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


rng = np.random.default_rng(28)
features = rng.normal(size=(700, 5))
logits = (
    1.3 * features[:, 0]
    - 0.9 * features[:, 1]
    + 0.7 * features[:, 2] * features[:, 3]
    - 0.2
)
probability = 1 / (1 + np.exp(-logits))
labels = (rng.random(len(features)) < probability).astype(float)
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.25, random_state=28, stratify=labels
)
scaler = StandardScaler().fit(x_train)
x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)


def activation(z, name):
    if name == "relu":
        return np.maximum(0, z)
    if name == "sigmoid":
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    return np.tanh(z)


def activation_derivative(z, name):
    if name == "relu":
        return (z > 0).astype(float)
    if name == "sigmoid":
        value = activation(z, name)
        return value * (1 - value)
    return 1 - np.tanh(z) ** 2


for name in ("relu", "sigmoid", "tanh"):
    model_rng = np.random.default_rng(280)
    w1 = model_rng.normal(0, np.sqrt(2 / x_train.shape[1]), (x_train.shape[1], 12))
    b1 = np.zeros((1, 12))
    w2 = model_rng.normal(0, 0.2, (12, 1))
    b2 = np.zeros((1, 1))
    target = y_train.reshape(-1, 1)
    for _ in range(500):
        z1 = x_train @ w1 + b1
        hidden = activation(z1, name)
        z2 = hidden @ w2 + b2
        predicted = 1 / (1 + np.exp(-np.clip(z2, -30, 30)))
        delta2 = (predicted - target) / len(target)
        grad_w2 = hidden.T @ delta2
        grad_b2 = delta2.sum(axis=0, keepdims=True)
        delta1 = (delta2 @ w2.T) * activation_derivative(z1, name)
        grad_w1 = x_train.T @ delta1
        grad_b1 = delta1.sum(axis=0, keepdims=True)
        w1 -= 0.05 * grad_w1
        b1 -= 0.05 * grad_b1
        w2 -= 0.05 * grad_w2
        b2 -= 0.05 * grad_b2

    test_hidden = activation(x_test @ w1 + b1, name)
    test_probability = 1 / (
        1 + np.exp(-np.clip(test_hidden @ w2 + b2, -30, 30))
    )
    test_probability = test_probability.ravel()
    print(
        f"{name:7}: test accuracy={accuracy_score(y_test, test_probability >= 0.5):.3f}, "
        f"log loss={log_loss(y_test, test_probability):.3f}"
    )

print("Activation effects depend on initialization, data, and learning settings; compare on the same split.")

"""Experiment with learning rates for a small neural demand regressor."""

import numpy as np


x = np.linspace(-2, 2, 80).reshape(-1, 1)
y = (1.5 * x[:, 0] + np.sin(2 * x[:, 0])).reshape(-1, 1)
x = (x - x.mean()) / x.std()
y = (y - y.mean()) / y.std()


def train(learning_rate):
    model_rng = np.random.default_rng(2020)
    weights_1 = model_rng.normal(0, 0.4, (1, 8))
    bias_1 = np.zeros((1, 8))
    weights_2 = model_rng.normal(0, 0.4, (8, 1))
    bias_2 = np.zeros((1, 1))
    losses = []
    for _ in range(600):
        hidden = np.tanh(x @ weights_1 + bias_1)
        prediction = hidden @ weights_2 + bias_2
        error = prediction - y
        loss = float(np.mean(error**2))
        losses.append(loss)
        output_gradient = 2 * error / len(x)
        weights_2_gradient = hidden.T @ output_gradient
        bias_2_gradient = output_gradient.sum(axis=0, keepdims=True)
        hidden_gradient = (output_gradient @ weights_2.T) * (1 - hidden**2)
        weights_1_gradient = x.T @ hidden_gradient
        bias_1_gradient = hidden_gradient.sum(axis=0, keepdims=True)
        weights_1 -= learning_rate * weights_1_gradient
        bias_1 -= learning_rate * bias_1_gradient
        weights_2 -= learning_rate * weights_2_gradient
        bias_2 -= learning_rate * bias_2_gradient
        if not np.isfinite(loss) or loss > 1e6:
            return losses[0], float("inf"), None
    return losses[0], losses[-1], losses


for rate in (0.001, 0.03, 1.0):
    initial, final, history = train(rate)
    if history is None:
        print(f"Learning rate {rate}: diverged; loss became unstable.")
    else:
        print(
            f"Learning rate {rate}: loss {initial:.4f} -> {final:.4f}; "
            f"loss after 50 epochs={history[49]:.4f}"
        )

print("A small rate learns slowly; a suitable rate converges faster; an excessive rate may oscillate or diverge.")

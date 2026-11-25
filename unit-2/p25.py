"""Train a feedforward neural network to learn the XOR truth table."""

import numpy as np


def sigmoid(values):
    return 1 / (1 + np.exp(-values))


inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
targets = np.array([[0], [1], [1], [0]], dtype=float)
rng = np.random.default_rng(25)
weights_hidden = rng.normal(0, 1, (2, 4))
bias_hidden = np.zeros((1, 4))
weights_output = rng.normal(0, 1, (4, 1))
bias_output = np.zeros((1, 1))

for _ in range(20_000):
    hidden = sigmoid(inputs @ weights_hidden + bias_hidden)
    output = sigmoid(hidden @ weights_output + bias_output)
    output_delta = (output - targets) * output * (1 - output)
    hidden_delta = (output_delta @ weights_output.T) * hidden * (1 - hidden)
    weights_output -= 1.0 * hidden.T @ output_delta / len(inputs)
    bias_output -= 1.0 * output_delta.mean(axis=0, keepdims=True)
    weights_hidden -= 1.0 * inputs.T @ hidden_delta / len(inputs)
    bias_hidden -= 1.0 * hidden_delta.mean(axis=0, keepdims=True)

predictions = sigmoid(sigmoid(inputs @ weights_hidden + bias_hidden) @ weights_output + bias_output)
for features, probability in zip(inputs.astype(int), predictions[:, 0]):
    print(f"{features.tolist()} XOR -> probability {probability:.3f}, class {int(probability >= 0.5)}")
print("A hidden layer provides the nonlinear representation needed to solve XOR.")

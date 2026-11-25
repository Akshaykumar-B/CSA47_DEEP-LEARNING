"""Calculate backpropagation gradients and update a small network."""

import numpy as np


def sigmoid(values):
    return 1 / (1 + np.exp(-values))


features = np.array([[0.7, 0.2]])
target = np.array([[1.0]])
weights_hidden = np.array([[0.1, -0.2], [0.4, 0.2]])
bias_hidden = np.array([[0.0, 0.1]])
weights_output = np.array([[0.3], [-0.1]])
bias_output = np.array([[0.05]])
learning_rate = 0.5

z_hidden = features @ weights_hidden + bias_hidden
hidden = sigmoid(z_hidden)
z_output = hidden @ weights_output + bias_output
prediction = sigmoid(z_output)

# Binary cross-entropy with sigmoid output has output delta = prediction - target.
delta_output = prediction - target
gradient_output = hidden.T @ delta_output
gradient_bias_output = delta_output
delta_hidden = (delta_output @ weights_output.T) * hidden * (1 - hidden)
gradient_hidden = features.T @ delta_hidden
gradient_bias_hidden = delta_hidden

print("Prediction before update:", float(prediction[0, 0]))
print("Output weight gradient:\n", gradient_output)
print("Hidden weight gradient:\n", gradient_hidden)
weights_output -= learning_rate * gradient_output
bias_output -= learning_rate * gradient_bias_output
weights_hidden -= learning_rate * gradient_hidden
bias_hidden -= learning_rate * gradient_bias_hidden

new_hidden = sigmoid(features @ weights_hidden + bias_hidden)
new_prediction = sigmoid(new_hidden @ weights_output + bias_output)
print("Prediction after update:", float(new_prediction[0, 0]))
print("The example shows one gradient step; repeated updates over data train the network.")

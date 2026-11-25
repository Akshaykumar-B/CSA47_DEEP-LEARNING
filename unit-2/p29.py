"""Perform forward propagation through a small facial-ID network."""

import numpy as np


def relu(values):
    return np.maximum(0, values)


def softmax(values):
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


input_data = np.array([0.6, 0.1, 0.8])
weights_hidden = np.array([[0.2, -0.4], [0.7, 0.3], [-0.5, 0.8]])
bias_hidden = np.array([0.1, 0.2])
weights_output = np.array([[0.4, -0.3, 0.5], [0.6, 0.2, -0.4]])
bias_output = np.array([0.0, 0.1, -0.1])

hidden_pre_activation = input_data @ weights_hidden + bias_hidden
hidden_output = relu(hidden_pre_activation)
output_logits = hidden_output @ weights_output + bias_output
class_probabilities = softmax(output_logits)

print("Input:", input_data)
print("Hidden pre-activation:", np.round(hidden_pre_activation, 3))
print("Hidden layer output (ReLU):", np.round(hidden_output, 3))
print("Output logits:", np.round(output_logits, 3))
print("Output probabilities:", np.round(class_probabilities, 3))
print("Predicted identity class:", int(np.argmax(class_probabilities)))

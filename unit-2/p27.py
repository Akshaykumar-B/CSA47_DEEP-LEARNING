"""Plot and compare sigmoid, tanh, and ReLU activation functions."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(-6, 6, 400)
activations = {
    "Sigmoid": 1 / (1 + np.exp(-x)),
    "Tanh": np.tanh(x),
    "ReLU": np.maximum(0, x),
}

activation_functions = {
    "Sigmoid": lambda value: 1 / (1 + np.exp(-value)),
    "Tanh": np.tanh,
    "ReLU": lambda value: np.maximum(0, value),
}
for name, function in activation_functions.items():
    print(f"{name}: f(-2)={float(function(-2)):.3f}, f(0)={float(function(0)):.3f}, f(2)={float(function(2)):.3f}")

figure, axis = plt.subplots(figsize=(8, 5))
for name, values in activations.items():
    axis.plot(x, values, label=name)
axis.axhline(0, color="gray", linewidth=0.7)
axis.axvline(0, color="gray", linewidth=0.7)
axis.set(title="Activation functions", xlabel="Input", ylabel="Output")
axis.grid(True, alpha=0.3)
axis.legend()
figure.tight_layout()
output_path = Path(__file__).with_name("p27_activations.png")
figure.savefig(output_path, dpi=150)
print(f"Plot saved to {output_path}")
print("Sigmoid outputs (0, 1); tanh outputs (-1, 1); ReLU is zero for negative input and linear for positive input.")

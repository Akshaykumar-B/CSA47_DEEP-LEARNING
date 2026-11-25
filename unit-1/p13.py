"""KL divergence between actual and predicted word distributions."""

from math import log2


def kl_divergence(actual, predicted):
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Distributions must have the same nonzero length")
    if any(value < 0 for value in actual + predicted):
        raise ValueError("Probabilities cannot be negative")
    if abs(sum(actual) - 1) > 1e-9 or abs(sum(predicted) - 1) > 1e-9:
        raise ValueError("Each distribution must sum to 1")
    if any(p > 0 and q == 0 for p, q in zip(actual, predicted)):
        return float("inf")
    return sum(p * log2(p / q) for p, q in zip(actual, predicted) if p > 0)


actual = [0.50, 0.30, 0.20]
predicted = [0.40, 0.40, 0.20]
divergence = kl_divergence(actual, predicted)
print("Actual word probabilities:", actual)
print("Predicted word probabilities:", predicted)
print(f"KL(actual || predicted): {divergence:.4f} bits")
print("KL divergence is nonnegative; larger values indicate more information mismatch.")

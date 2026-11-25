"""Binary cross-entropy loss for spam predictions."""

from math import log


def binary_cross_entropy(actual, predicted):
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Actual and predicted values must have the same nonzero length")
    epsilon = 1e-15
    losses = []
    for label, probability in zip(actual, predicted):
        if label not in (0, 1) or not 0 <= probability <= 1:
            raise ValueError("Labels must be 0/1 and probabilities must be in [0, 1]")
        probability = min(max(probability, epsilon), 1 - epsilon)
        losses.append(
            -(label * log(probability) + (1 - label) * log(1 - probability))
        )
    return losses


actual = [1, 0, 1, 0]
predicted = [0.90, 0.20, 0.40, 0.70]
losses = binary_cross_entropy(actual, predicted)
for index, loss in enumerate(losses, start=1):
    print(f"Email {index}: cross-entropy loss = {loss:.3f}")
print(f"Mean cross-entropy loss: {sum(losses) / len(losses):.3f}")
print("The confident wrong predictions (emails 3 and 4) incur the largest penalties.")

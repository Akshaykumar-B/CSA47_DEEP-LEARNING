"""Calculate classification entropy for loan applicants."""

from math import log2


def entropy(class_counts):
    total = sum(class_counts)
    if total == 0:
        return 0.0
    return -sum(
        (count / total) * log2(count / total)
        for count in class_counts
        if count > 0
    )


low_risk, high_risk = 6, 4
uncertainty = entropy([low_risk, high_risk])
print(f"Applicants: {low_risk} low-risk, {high_risk} high-risk")
print(f"Dataset entropy: {uncertainty:.3f} bits (maximum for two classes is 1 bit)")
print("Entropy near 1 indicates substantial class uncertainty; entropy near 0 indicates a dominant class.")

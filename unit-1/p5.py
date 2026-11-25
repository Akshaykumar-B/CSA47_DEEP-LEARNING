"""Dot product and cosine similarity for customer preferences."""

from math import sqrt


def dot_product(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    return sum(x * y for x, y in zip(a, b))


def cosine_similarity(a, b):
    denominator = sqrt(dot_product(a, a) * dot_product(b, b))
    if denominator == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector")
    return dot_product(a, b) / denominator


customer_a = [5, 1, 0, 2]
customer_b = [4, 1, 1, 2]
customer_c = [0, 4, 5, 1]

for name, preferences in (("B", customer_b), ("C", customer_c)):
    print(f"Dot product A/{name}:", dot_product(customer_a, preferences))
    print(f"Cosine similarity A/{name}:", f"{cosine_similarity(customer_a, preferences):.3f}")

closest = max(
    (("B", customer_b), ("C", customer_c)),
    key=lambda item: cosine_similarity(customer_a, item[1]),
)
print("Customer A has more similar interests to customer", closest[0])

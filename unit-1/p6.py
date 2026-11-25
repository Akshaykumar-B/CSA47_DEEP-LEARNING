"""Compare Euclidean and Manhattan distances for customer behavior."""

from math import sqrt


def euclidean(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def manhattan(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    return sum(abs(x - y) for x, y in zip(a, b))


customer_a = [0, 0]
customer_b = [3, 4]
customer_c = [2, 2]

for name, customer in (("B", customer_b), ("C", customer_c)):
    print(
        f"A to {name}: Euclidean={euclidean(customer_a, customer):.3f}, "
        f"Manhattan={manhattan(customer_a, customer):.3f}"
    )
print("Both metrics rank C as closer to A in this example.")
print("Manhattan is axis-aligned and less sensitive to a single large difference.")
print("Euclidean measures straight-line distance; select based on the feature meaning.")

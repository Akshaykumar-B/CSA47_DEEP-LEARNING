"""A two-page transition matrix and its eigenpairs."""

from math import sqrt


def eigenvalues_2x2(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    trace = a + d
    determinant = a * d - b * c
    discriminant = sqrt(trace * trace - 4 * determinant)
    return [(trace + discriminant) / 2, (trace - discriminant) / 2]


def eigenvector_2x2(matrix, eigenvalue):
    a, b = matrix[0]
    vector = [b, eigenvalue - a]
    length = sqrt(sum(value * value for value in vector))
    if length == 0:
        raise ValueError("Unable to determine an eigenvector for this matrix")
    return [value / length for value in vector]


# Columns represent the page being left; each column sums to one.
transition = [[0.5, 0.25], [0.5, 0.75]]
values = eigenvalues_2x2(transition)

print("Transition matrix (columns are source pages):", transition)
for value in values:
    print(f"Eigenvalue {value:.3f}: eigenvector {eigenvector_2x2(transition, value)}")
dominant = eigenvector_2x2(transition, values[0])
scores = [abs(value) for value in dominant]
total = sum(scores)
print("Normalized page importance:", [score / total for score in scores])
print("The eigenvector for eigenvalue 1 gives the long-run page importance.")

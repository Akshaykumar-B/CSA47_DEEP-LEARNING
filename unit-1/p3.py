"""Determinant, inverse, and rank of a small encryption matrix."""


def determinant(matrix):
    work = [row[:] for row in matrix]
    n = len(work)
    if any(len(row) != n for row in work):
        raise ValueError("Determinant requires a square matrix")
    result = 1.0
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) < 1e-12:
            return 0.0
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            result *= -1
        value = work[column][column]
        result *= value
        for row in range(column + 1, n):
            factor = work[row][column] / value
            for j in range(column + 1, n):
                work[row][j] -= factor * work[column][j]
    return result


def inverse(matrix):
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("Inverse requires a square matrix")
    work = [
        [float(value) for value in row]
        + [float(i == j) for j in range(n)]
        for i, row in enumerate(matrix)
    ]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) < 1e-12:
            raise ValueError("Matrix is singular and has no inverse")
        work[column], work[pivot] = work[pivot], work[column]
        divisor = work[column][column]
        work[column] = [value / divisor for value in work[column]]
        for row in range(n):
            if row != column:
                factor = work[row][column]
                work[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(work[row], work[column])
                ]
    return [row[n:] for row in work]


def rank(matrix):
    work = [[float(value) for value in row] for row in matrix]
    rows, columns = len(work), len(work[0])
    pivot_row = 0
    for column in range(columns):
        pivot = max(range(pivot_row, rows), key=lambda row: abs(work[row][column]), default=pivot_row)
        if pivot_row == rows or abs(work[pivot][column]) < 1e-12:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        divisor = work[pivot_row][column]
        work[pivot_row] = [value / divisor for value in work[pivot_row]]
        for row in range(rows):
            if row != pivot_row:
                factor = work[row][column]
                work[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(work[row], work[pivot_row])
                ]
        pivot_row += 1
        if pivot_row == rows:
            break
    return pivot_row


key_matrix = [[2, 3], [3, 5]]
det = determinant(key_matrix)
matrix_rank = rank(key_matrix)
print("Encryption matrix:", key_matrix)
print("Determinant:", det)
print("Inverse:", inverse(key_matrix) if abs(det) > 1e-12 else "not defined")
print("Rank:", matrix_rank)
print(
    "Suitable as an invertible linear key:",
    abs(det) > 1e-12 and matrix_rank == len(key_matrix),
)
print("Note: a real cipher also needs secure key handling and a suitable cryptographic design.")

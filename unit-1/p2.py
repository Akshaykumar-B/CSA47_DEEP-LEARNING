"""Matrix multiplication and transpose for image-coordinate transforms."""


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def multiply(a, b):
    if not a or not b or len(a[0]) != len(b):
        raise ValueError("Incompatible matrix dimensions")
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


scale = [[2, 0], [0, 3]]
rotate_90 = [[0, -1], [1, 0]]
combined = multiply(rotate_90, scale)

print("Scale matrix:", scale)
print("Rotation matrix:", rotate_90)
print("Rotation * scale:", combined)
print("Transpose of result:", transpose(combined))
print("Scale * rotation:", multiply(scale, rotate_90))
print("Analysis: multiplication order changes the transform; transpose swaps rows and columns.")

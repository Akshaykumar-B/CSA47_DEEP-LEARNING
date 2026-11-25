"""3D vector operations for a robot position."""


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def subtract(a, b):
    return [x - y for x, y in zip(a, b)]


def scalar_multiply(vector, scalar):
    return [scalar * value for value in vector]


position = [2, 1, 4]
movement = [3, -2, 1]

print("Current position:", position)
print("Movement vector:", movement)
print("New position:", add(position, movement))
print("Position minus movement:", subtract(position, movement))
print("Twice the movement:", scalar_multiply(movement, 2))

"""Minimize a quadratic production-cost function with gradient descent."""


def cost(parameter):
    return (parameter - 3) ** 2 + 2


def gradient(parameter):
    return 2 * (parameter - 3)


parameter = -4.0
learning_rate = 0.1
iterations = 50

for _ in range(iterations):
    parameter -= learning_rate * gradient(parameter)

print(f"Minimum parameter estimate: {parameter:.6f}")
print(f"Cost at estimate: {cost(parameter):.6f}")
print("Analytical minimum: parameter=3, cost=2")

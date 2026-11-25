"""Compare batch, stochastic, and mini-batch gradient descent."""

from random import Random


data = [(x / 10, 2 * (x / 10) + 1) for x in range(-20, 21)]
epochs = 100
learning_rate = 0.01
batch_size = 5


def loss(parameters):
    weight, bias = parameters
    return sum((weight * x + bias - y) ** 2 for x, y in data) / len(data)


def update(parameters, batch):
    weight, bias = parameters
    count = len(batch)
    weight_gradient = sum(2 * (weight * x + bias - y) * x for x, y in batch) / count
    bias_gradient = sum(2 * (weight * x + bias - y) for x, y in batch) / count
    return [
        weight - learning_rate * weight_gradient,
        bias - learning_rate * bias_gradient,
    ]


batch_parameters = [0.0, 0.0]
for _ in range(epochs):
    batch_parameters = update(batch_parameters, data)

rng = Random(7)
stochastic_parameters = [0.0, 0.0]
for _ in range(epochs):
    shuffled = data[:]
    rng.shuffle(shuffled)
    for sample in shuffled:
        stochastic_parameters = update(stochastic_parameters, [sample])

mini_parameters = [0.0, 0.0]
for _ in range(epochs):
    shuffled = data[:]
    rng.shuffle(shuffled)
    for start in range(0, len(shuffled), batch_size):
        mini_parameters = update(mini_parameters, shuffled[start : start + batch_size])

methods = (
    ("Batch", batch_parameters, epochs),
    ("Stochastic", stochastic_parameters, epochs * len(data)),
    ("Mini-batch", mini_parameters, epochs * ((len(data) + batch_size - 1) // batch_size)),
)
for name, parameters, updates in methods:
    print(
        f"{name:10} weight={parameters[0]:.4f}, bias={parameters[1]:.4f}, "
        f"MSE={loss(parameters):.6f}, updates={updates}"
    )
print("Batch uses fewer, stable updates; stochastic updates are noisy; mini-batches trade off both.")
print("All methods process the same number of examples per epoch, but update costs differ.")

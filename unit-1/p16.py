"""Predict house prices with multiple linear regression and gradient descent."""


def predict(features, parameters):
    return sum(feature * parameter for feature, parameter in zip([1.0] + features, parameters))


training = [
    ([0.8, 2, 0], 200),
    ([1.0, 2, 0], 230),
    ([1.2, 3, 0], 280),
    ([1.4, 3, 1], 350),
    ([1.6, 4, 1], 400),
    ([1.8, 4, 1], 430),
    ([2.0, 5, 1], 500),
    ([1.1, 2, 1], 300),
]
testing = [
    ([1.5, 3, 1], 375),
    ([0.9, 2, 0], 215),
]
# Features are area in thousands of square feet, bedrooms, and a location indicator.
parameters = [0.0, 0.0, 0.0, 0.0]
learning_rate = 0.01

for _ in range(20_000):
    gradients = [0.0] * len(parameters)
    for features, actual in training:
        difference = predict(features, parameters) - actual
        for index, feature in enumerate([1.0] + features):
            gradients[index] += 2 * difference * feature / len(training)
    parameters = [
        parameter - learning_rate * gradient
        for parameter, gradient in zip(parameters, gradients)
    ]

errors = []
for features, actual in testing:
    estimate = predict(features, parameters)
    error = estimate - actual
    errors.append(error)
    print(f"Actual=${actual:.1f}k, predicted=${estimate:.1f}k, error=${error:.1f}k")

mae = sum(abs(error) for error in errors) / len(errors)
mse = sum(error * error for error in errors) / len(errors)
print("Learned parameters [intercept, area, bedrooms, location]:", [round(x, 2) for x in parameters])
print(f"Test MAE: ${mae:.2f}k")
print(f"Test RMSE: ${(mse ** 0.5):.2f}k")

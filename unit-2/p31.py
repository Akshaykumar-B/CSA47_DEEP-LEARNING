"""Train a neural house-price regressor with mean squared error."""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


rng = np.random.default_rng(31)
features = rng.uniform([0.6, 1, 0], [2.5, 5, 1], size=(240, 3))
# Features: area (thousand square feet), bedrooms, and a location indicator.
prices = (
    45
    + 125 * features[:, 0]
    + 18 * features[:, 1]
    + 55 * features[:, 2]
    + rng.normal(0, 15, len(features))
)
x_train, x_test, y_train, y_test = train_test_split(
    features, prices, test_size=0.25, random_state=31
)
model = make_pipeline(
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(24, 12),
        activation="relu",
        solver="adam",
        learning_rate_init=0.01,
        max_iter=5000,
        random_state=31,
    ),
)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
for actual, predicted in zip(y_test[:8], predictions[:8]):
    print(f"Actual=${actual:.1f}k, predicted=${predicted:.1f}k, error=${predicted - actual:+.1f}k")
print(f"Test MSE (the training loss): {mean_squared_error(y_test, predictions):.2f} ($k)^2")
print(f"Test RMSE: {mean_squared_error(y_test, predictions) ** 0.5:.2f} ($k)")

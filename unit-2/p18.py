"""Compare polynomial regression degrees using training and test errors."""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression


rng = np.random.default_rng(18)
x = np.linspace(-3, 3, 100).reshape(-1, 1)
y = (0.5 * x[:, 0] ** 2 + np.sin(2 * x[:, 0]) + rng.normal(0, 0.7, len(x)))
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=18
)

print("degree | train MSE | test MSE")
for degree in (1, 2, 5, 15):
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        LinearRegression(),
    )
    model.fit(x_train, y_train)
    train_mse = mean_squared_error(y_train, model.predict(x_train))
    test_mse = mean_squared_error(y_test, model.predict(x_test))
    print(f"{degree:6} | {train_mse:9.3f} | {test_mse:8.3f}")

print("Low degree can underfit (high bias); a large train-test gap suggests overfitting (high variance).")
print("Choose a degree with low validation/test error and a modest train-test gap.")

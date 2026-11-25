"""Compare L1 and L2 regularization for a loan-amount regression model."""

import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


rng = np.random.default_rng(19)
features = rng.normal(size=(240, 6))
true_coefficients = np.array([4.0, 0.0, -2.5, 0.0, 1.5, 0.0])
amount = 50 + features @ true_coefficients + rng.normal(0, 1.2, len(features))
x_train, x_test, y_train, y_test = train_test_split(
    features, amount, test_size=0.25, random_state=19
)
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

for name, model in (
    ("L1 (Lasso)", Lasso(alpha=0.25, max_iter=20_000)),
    ("L2 (Ridge)", Ridge(alpha=2.0)),
):
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print(f"{name}: coefficients={np.round(model.coef_, 3).tolist()}")
    print(f"  Test MSE={mean_squared_error(y_test, prediction):.3f}")
    print(f"  Coefficients near zero={int(np.sum(np.abs(model.coef_) < 0.05))}")

print("L1 can set coefficients exactly to zero (feature selection); L2 usually shrinks them smoothly.")
print("The better test error depends on the data and regularization strength.")

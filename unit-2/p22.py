"""Compare grid, random, and Gaussian-process Bayesian hyperparameter search."""

import numpy as np
from scipy.stats import norm
from sklearn.datasets import load_breast_cancer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


features, labels = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.25, random_state=22, stratify=labels
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=22)


def make_model(c_value):
    return make_pipeline(StandardScaler(), SVC(C=float(c_value), gamma="scale"))


grid_values = [0.01, 0.1, 1.0, 10.0, 100.0]
grid_scores = {
    value: cross_val_score(make_model(value), x_train, y_train, cv=cv, scoring="accuracy").mean()
    for value in grid_values
}
grid_best = max(grid_scores, key=grid_scores.get)

random_search = RandomizedSearchCV(
    make_pipeline(StandardScaler(), SVC(gamma="scale")),
    param_distributions={"svc__C": np.logspace(-2, 2, 20)},
    n_iter=5,
    cv=cv,
    scoring="accuracy",
    random_state=22,
)
random_search.fit(x_train, y_train)
random_best = float(random_search.best_params_["svc__C"])

# Sequential Bayesian optimization over log10(C), using expected improvement.
candidate_log_c = np.linspace(-2, 2, 81)
observed_log_c = [-2.0, 0.0, 2.0]


def validation_score(log_c):
    return cross_val_score(
        make_model(10**log_c), x_train, y_train, cv=cv, scoring="accuracy"
    ).mean()


observed_scores = [validation_score(value) for value in observed_log_c]
for _ in range(4):
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5) + WhiteKernel(noise_level=1e-5, noise_level_bounds="fixed"),
        normalize_y=True,
        n_restarts_optimizer=1,
        random_state=22,
    )
    gp.fit(np.array(observed_log_c).reshape(-1, 1), np.array(observed_scores))
    mean, std = gp.predict(candidate_log_c.reshape(-1, 1), return_std=True)
    improvement = mean - max(observed_scores) - 0.01
    expected_improvement = improvement * norm.cdf(improvement / std) + std * norm.pdf(
        improvement / std
    )
    expected_improvement[ np.isin(candidate_log_c, observed_log_c) ] = -np.inf
    next_log_c = float(candidate_log_c[np.argmax(expected_improvement)])
    observed_log_c.append(next_log_c)
    observed_scores.append(validation_score(next_log_c))
bayes_best = 10 ** observed_log_c[int(np.argmax(observed_scores))]

for name, best_c in (
    ("Grid search", grid_best),
    ("Random search", random_best),
    ("Bayesian optimization", bayes_best),
):
    model = make_model(best_c).fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    trials = (
        len(grid_values)
        if name == "Grid search"
        else 5
        if name == "Random search"
        else len(observed_log_c)
    )
    print(
        f"{name}: selected C={best_c:.4g}, held-out test accuracy={accuracy:.3f}, "
        f"configurations evaluated={trials}"
    )

print("Grid search is exhaustive over its small grid; random search samples broadly.")
print("Bayesian optimization uses prior trial results to choose promising next values.")

"""Compare training and testing errors to diagnose neural-network overfitting."""

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


features, labels = make_classification(
    n_samples=300,
    n_features=40,
    n_informative=8,
    n_redundant=8,
    n_classes=2,
    random_state=17,
)
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=17, stratify=labels
)

models = {
    "Large network (baseline)": MLPClassifier(
        hidden_layer_sizes=(100, 100),
        alpha=1e-5,
        max_iter=500,
        random_state=17,
    ),
    "Regularized smaller network": MLPClassifier(
        hidden_layer_sizes=(10,),
        alpha=1.0,
        max_iter=1000,
        random_state=17,
    ),
}

for name, classifier in models.items():
    model = make_pipeline(StandardScaler(), classifier)
    model.fit(x_train, y_train)
    train_error = 1 - accuracy_score(y_train, model.predict(x_train))
    test_error = 1 - accuracy_score(y_test, model.predict(x_test))
    print(f"{name}: training error={train_error:.3f}, testing error={test_error:.3f}")
    if test_error > train_error + 0.1:
        print("  Large generalization gap: evidence of overfitting.")
    else:
        print("  Smaller train-test gap: better generalization on this split.")

print("Other remedies: use more representative data, feature selection, dropout, or cross-validation.")

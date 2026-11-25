"""Classify Iris species using a multilayer neural network."""

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.25,
    random_state=26,
    stratify=iris.target,
)
model = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(12, 6), max_iter=1500, random_state=26),
)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
print("Species:", iris.target_names.tolist())
print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
print(
    classification_report(
        y_test, predictions, target_names=iris.target_names, zero_division=0
    )
)

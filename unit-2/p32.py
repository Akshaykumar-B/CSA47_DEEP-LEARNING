"""Evaluate a binary neural network for fraudulent-transaction detection."""

from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


features, labels = make_classification(
    n_samples=1200,
    n_features=12,
    n_informative=7,
    n_redundant=3,
    weights=[0.92, 0.08],
    class_sep=1.2,
    random_state=32,
)
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.25, random_state=32, stratify=labels
)
model = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(24, 12),
        alpha=0.01,
        max_iter=1000,
        early_stopping=True,
        random_state=32,
    ),
)
model.fit(x_train, y_train)
probabilities = model.predict_proba(x_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

print(f"Binary cross-entropy: {log_loss(y_test, probabilities):.3f}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.3f}")
print(f"Recall: {recall_score(y_test, predictions, zero_division=0):.3f}")
print(f"F1 score: {f1_score(y_test, predictions, zero_division=0):.3f}")
print(f"ROC AUC: {roc_auc_score(y_test, probabilities):.3f}")
print("Confusion matrix [ [true legitimate, false fraud], [missed fraud, detected fraud] ]:")
print(confusion_matrix(y_test, predictions))
print("For imbalanced fraud data, recall, precision, and threshold selection matter beyond accuracy.")

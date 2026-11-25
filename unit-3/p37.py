"""Compare regularized and unregularized patient-risk classifiers."""

import torch
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn


torch.manual_seed(37)
torch.set_num_threads(1)
features, labels = make_classification(
    n_samples=1_000, n_features=14, n_informative=8, n_redundant=3,
    class_sep=0.85, random_state=37
)
features = StandardScaler().fit_transform(features)
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.25, random_state=37, stratify=labels
)
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

for name, weight_decay in (("without regularization", 0.0), ("with L2 regularization", 0.03)):
    model = nn.Sequential(nn.Linear(14, 40), nn.ReLU(), nn.Linear(40, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
    for _ in range(40):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x_train), y_train_tensor)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_predictions = model(x_train).argmax(1).numpy()
        test_predictions = model(x_test).argmax(1).numpy()
    print(
        f"{name:24}: train accuracy={accuracy_score(y_train, train_predictions):.3f}, "
        f"test accuracy={accuracy_score(y_test, test_predictions):.3f}, "
        f"test F1={f1_score(y_test, test_predictions):.3f}"
    )
print("The model with better held-out metrics is the better-generalizing model on this split.")

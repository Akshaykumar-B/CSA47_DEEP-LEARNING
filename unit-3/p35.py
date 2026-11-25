"""Evaluate dropout regularization on synthetic student-performance data."""

import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn


torch.manual_seed(35)
torch.set_num_threads(1)
features, labels = make_classification(
    n_samples=900, n_features=20, n_informative=8, n_redundant=6,
    class_sep=0.9, random_state=35
)
features = StandardScaler().fit_transform(features)
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.25, random_state=35, stratify=labels
)
tensors = [
    torch.tensor(x, dtype=torch.float32)
    for x in (x_train, y_train, x_test, y_test)
]
x_train, y_train, x_test, y_test = tensors
y_train, y_test = y_train.long(), y_test.long()


def build_model(dropout):
    layers = [nn.Linear(20, 64), nn.ReLU()]
    if dropout:
        layers.append(nn.Dropout(0.35))
    layers.extend([nn.Linear(64, 32), nn.ReLU()])
    if dropout:
        layers.append(nn.Dropout(0.35))
    layers.append(nn.Linear(32, 2))
    return nn.Sequential(*layers)


for name, dropout in (("without dropout", 0.0), ("with dropout", 0.35)):
    model = build_model(dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(45):
        model.train()
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_accuracy = (model(x_train).argmax(1) == y_train).float().mean().item()
        test_accuracy = (model(x_test).argmax(1) == y_test).float().mean().item()
    print(f"{name:18}: train accuracy={train_accuracy:.3f}, test accuracy={test_accuracy:.3f}")
print("A smaller train-test gap can indicate improved generalization; compare both scores.")

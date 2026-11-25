"""Measure how L2 regularization affects weights and test performance."""

import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn


torch.manual_seed(36)
torch.set_num_threads(1)
features, labels = make_classification(
    n_samples=1_000, n_features=18, n_informative=9, n_redundant=5,
    class_sep=0.8, random_state=36
)
features = StandardScaler().fit_transform(features)
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.25, random_state=36, stratify=labels
)
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

for label, weight_decay in (("no L2", 0.0), ("L2 (weight_decay=0.05)", 0.05)):
    model = nn.Sequential(nn.Linear(18, 48), nn.ReLU(), nn.Linear(48, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
    for _ in range(50):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        accuracy = (model(x_test).argmax(1) == y_test).float().mean().item()
        weight_norm = torch.sqrt(sum(parameter.square().sum() for parameter in model.parameters())).item()
    print(f"{label:23}: test accuracy={accuracy:.3f}, L2 weight norm={weight_norm:.3f}")
print("L2 penalizes large weights; its effect on generalization should be judged on held-out data.")

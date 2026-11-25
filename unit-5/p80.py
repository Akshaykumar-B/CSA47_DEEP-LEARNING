"""Train and evaluate a medical-attribute classifier as a deep-learning demo."""

import torch
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(80)
torch.set_num_threads(1)
features, labels = make_classification(
    n_samples=1600,
    n_features=16,
    n_informative=9,
    n_redundant=4,
    class_sep=1.0,
    weights=[0.7, 0.3],
    random_state=80,
)
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.25, random_state=80, stratify=labels
)
scaler = StandardScaler().fit(x_train)
x_train = torch.tensor(scaler.transform(x_train), dtype=torch.float32)
x_test = torch.tensor(scaler.transform(x_test), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
model = nn.Sequential(
    nn.Linear(16, 48),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(48, 24),
    nn.ReLU(),
    nn.Linear(24, 2),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
for epoch in range(35):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_y)
    if epoch in (0, 9, 19, 34):
        print(f"Epoch {epoch + 1}: train loss={total_loss / len(train_loader.dataset):.4f}")
model.eval()
with torch.no_grad():
    predictions = model(x_test).argmax(1).numpy()
actual = y_test.numpy()
print(f"Test accuracy: {accuracy_score(actual, predictions):.3f}")
print("Confusion matrix [class 0, class 1]:\n", confusion_matrix(actual, predictions))
print(classification_report(actual, predictions, target_names=["lower risk", "higher risk"], zero_division=0))
print("Synthetic demonstration only; do not use for real clinical decisions.")

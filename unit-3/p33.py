"""Compare SGD, RMSprop, and Adam on a synthetic product-defect dataset."""

import time

import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(33)
torch.set_num_threads(1)
features, labels = make_classification(
    n_samples=900, n_features=16, n_informative=10, n_redundant=4,
    class_sep=1.1, random_state=33
)
scaler = StandardScaler().fit(features)
features = scaler.transform(features)
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.25, random_state=33, stratify=labels
)
train_data = TensorDataset(
    torch.tensor(x_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long),
)
test_x = torch.tensor(x_test, dtype=torch.float32)
test_y = torch.tensor(y_test, dtype=torch.long)


def make_model():
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))


for name in ("SGD", "RMSprop", "Adam"):
    model = make_model()
    optimizer_type = getattr(torch.optim, name)
    optimizer = optimizer_type(model.parameters(), lr=0.03 if name == "SGD" else 0.01)
    loader = DataLoader(train_data, batch_size=64, shuffle=True)
    started = time.perf_counter()
    for _ in range(18):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    elapsed = time.perf_counter() - started
    with torch.no_grad():
        predictions = model(test_x).argmax(dim=1)
        accuracy = (predictions == test_y).float().mean().item()
        final_loss = nn.functional.cross_entropy(model(test_x), test_y).item()
    print(
        f"{name:7} | test accuracy={accuracy:.3f} | "
        f"test loss={final_loss:.3f} | training time={elapsed:.2f}s"
    )
print("Convergence time and accuracy depend on the data, learning rate, and hardware.")

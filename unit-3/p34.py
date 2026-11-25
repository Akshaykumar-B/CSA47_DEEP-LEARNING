"""Train a deep feedforward network on a reproducible MNIST subset."""

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


torch.manual_seed(34)
torch.set_num_threads(1)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_set = datasets.MNIST("data", train=True, download=True, transform=transform)
test_set = datasets.MNIST("data", train=False, download=True, transform=transform)
train_loader = DataLoader(Subset(train_set, range(10_000)), batch_size=128, shuffle=True)
test_loader = DataLoader(Subset(test_set, range(2_000)), batch_size=256)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            correct += (model(images).argmax(1) == labels).sum().item()
            total += len(labels)
    print(f"Epoch {epoch + 1}: train loss={total_loss / len(train_loader.dataset):.4f}, test accuracy={correct / total:.3f}")

print("Evaluation uses 2,000 held-out MNIST images; the first run downloads MNIST.")

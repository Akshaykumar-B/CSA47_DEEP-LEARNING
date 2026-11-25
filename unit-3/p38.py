"""Train a convolutional neural network on a reproducible MNIST subset."""

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


torch.manual_seed(38)
torch.set_num_threads(1)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_set = datasets.MNIST("data", train=True, download=True, transform=transform)
test_set = datasets.MNIST("data", train=False, download=True, transform=transform)
train_loader = DataLoader(Subset(train_set, range(8_000)), batch_size=128, shuffle=True)
test_loader = DataLoader(Subset(test_set, range(2_000)), batch_size=256)
model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 7 * 7, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            correct += (model(images).argmax(1) == labels).sum().item()
            total += len(labels)
    print(f"Epoch {epoch + 1}: held-out accuracy={correct / total:.3f}")
print("The held-out set contains 2,000 MNIST digits; the first run downloads MNIST.")

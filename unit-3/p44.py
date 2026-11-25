"""Compare CNN training with and without batch normalization."""

import torch
from sklearn.model_selection import train_test_split
from torch import nn


def make_images(count, seed):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(count, 1, 16, 16, generator=generator) * 0.15
    labels = torch.randint(0, 2, (count,), generator=generator)
    for index, label in enumerate(labels):
        if label:
            images[index, 0, 5:11, 5:11] += 0.8
    return images, labels


torch.manual_seed(44)
torch.set_num_threads(1)
images, labels = make_images(700, 44)
train_indices, test_indices = train_test_split(
    range(len(labels)), test_size=0.25, random_state=44, stratify=labels.numpy()
)
x_train, y_train = images[train_indices], labels[train_indices]
x_test, y_test = images[test_indices], labels[test_indices]


def make_model(batch_norm):
    layers = [nn.Conv2d(1, 16, 3, padding=1)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(16))
    layers.extend([nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(16 * 8 * 8, 2)])
    return nn.Sequential(*layers)


for label, enabled in (("without batch norm", False), ("with batch norm", True)):
    model = make_model(enabled)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(12):
        model.train()
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        if epoch in (0, 5, 11):
            print(f"{label:19}, epoch {epoch + 1:2}: train loss={loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        accuracy = (model(x_test).argmax(1) == y_test).float().mean().item()
    print(f"{label:19}: test accuracy={accuracy:.3f}")

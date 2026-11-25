"""Compare CNN filter sizes and filter counts on synthetic defect images."""

import torch
from sklearn.model_selection import train_test_split
from torch import nn


def make_images(count, seed):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(count, 1, 16, 16, generator=generator) * 0.12
    labels = torch.randint(0, 2, (count,), generator=generator)
    for index, label in enumerate(labels):
        if label:
            images[index, 0, 5:11, 5:11] += 0.8
    return images, labels


torch.manual_seed(42)
torch.set_num_threads(1)
images, labels = make_images(700, 42)
indices = torch.arange(len(labels)).numpy()
train_indices, test_indices = train_test_split(
    indices, test_size=0.25, random_state=42, stratify=labels.numpy()
)
x_train, y_train = images[train_indices], labels[train_indices]
x_test, y_test = images[test_indices], labels[test_indices]


def build_model(kernel_size, filters):
    return nn.Sequential(
        nn.Conv2d(1, filters, kernel_size, padding=kernel_size // 2),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(filters, 2),
    )


for kernel_size, filters in ((3, 8), (5, 8), (3, 16), (5, 16)):
    model = build_model(kernel_size, filters)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(12):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        accuracy = (model(x_test).argmax(1) == y_test).float().mean().item()
    print(f"kernel={kernel_size}x{kernel_size}, filters={filters:2}: test accuracy={accuracy:.3f}")
print("Larger kernels cover wider regions; more filters represent more feature types.")

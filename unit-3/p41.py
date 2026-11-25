"""Train a cat-versus-dog CNN from a labelled ImageFolder dataset."""

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="folder with cats/ and dogs/ subfolders")
    args = parser.parse_args()
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    if len(dataset.classes) != 2 or len(dataset) < 4:
        raise ValueError("data_dir must contain exactly two class folders with at least four images")
    train_size = int(0.8 * len(dataset))
    train_set, test_set = random_split(
        dataset, [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(41),
    )
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(64, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5):
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
        print(f"Epoch {epoch + 1}: test accuracy={correct / total:.3f}")
    print("Class order:", dataset.classes)


if __name__ == "__main__":
    main()

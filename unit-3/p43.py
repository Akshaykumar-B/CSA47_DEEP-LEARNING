"""Evaluate a CNN on labelled normal/abnormal medical image folders."""

import argparse

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="folder containing normal/ and abnormal/ image subfolders")
    args = parser.parse_args()
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    if len(dataset.classes) != 2 or len(dataset) < 4:
        raise ValueError("data_dir must contain exactly two class folders and at least four images")
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=0.25,
        random_state=43,
        stratify=dataset.targets,
    )
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=32, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=32)
    model = nn.Sequential(
        nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(24, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} complete")
    model.eval()
    predictions, actual = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            predictions.extend(model(images).argmax(1).tolist())
            actual.extend(labels.tolist())
    print(f"Accuracy: {accuracy_score(actual, predictions):.3f}")
    print(f"Confusion matrix {dataset.classes}:\n", confusion_matrix(actual, predictions))
    print(classification_report(actual, predictions, target_names=dataset.classes, zero_division=0))
    print("This educational model is not a clinical diagnostic tool.")


if __name__ == "__main__":
    main()

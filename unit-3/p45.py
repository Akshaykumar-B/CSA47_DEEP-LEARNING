"""Compare medical-image classification with and without training augmentation."""

import argparse

import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="folder with one labelled subfolder per image class")
    args = parser.parse_args()
    plain_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    augmented_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(64, scale=(0.85, 1.0)),
        transforms.ToTensor(),
    ])
    plain_dataset = datasets.ImageFolder(args.data_dir, transform=plain_transform)
    augmented_dataset = datasets.ImageFolder(args.data_dir, transform=augmented_transform)
    if len(plain_dataset.classes) < 2 or len(plain_dataset) < 4:
        raise ValueError("data_dir must contain at least two class folders and four images")
    train_indices, test_indices = train_test_split(
        range(len(plain_dataset)),
        test_size=0.25,
        random_state=45,
        stratify=plain_dataset.targets,
    )
    test_loader = DataLoader(Subset(plain_dataset, test_indices), batch_size=32)

    def make_model():
        return nn.Sequential(
            nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(24, len(plain_dataset.classes)),
        )

    for label, dataset in (
        ("without augmentation", plain_dataset),
        ("with augmentation", augmented_dataset),
    ):
        train_loader = DataLoader(
            Subset(dataset, train_indices), batch_size=32, shuffle=True
        )
        model = make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(5):
            model.train()
            for images, classes in train_loader:
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(model(images), classes)
                loss.backward()
                optimizer.step()
        model.eval()
        predictions, actual = [], []
        with torch.no_grad():
            for images, classes in test_loader:
                predictions.extend(model(images).argmax(1).tolist())
                actual.extend(classes.tolist())
        print(f"{label}: held-out accuracy={accuracy_score(actual, predictions):.3f}")
    print("Classes:", plain_dataset.classes)


if __name__ == "__main__":
    main()

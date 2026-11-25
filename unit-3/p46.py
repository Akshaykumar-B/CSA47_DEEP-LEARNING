"""Fine-tune a pretrained ResNet-18 on a labelled plant-leaf image folder."""

import argparse

import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="folder with one subfolder per plant-disease class")
    args = parser.parse_args()
    weights = models.ResNet18_Weights.DEFAULT
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    if len(dataset.classes) < 2 or len(dataset) < 4:
        raise ValueError("data_dir must contain at least two class folders and four images")
    train_indices, test_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=46, stratify=dataset.targets
    )
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=16, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=16)
    model = models.resnet18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    for epoch in range(3):
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
    print(f"Test accuracy: {accuracy_score(actual, predictions):.3f}")
    print(classification_report(actual, predictions, target_names=dataset.classes, zero_division=0))
    print("The pretrained ImageNet weights are downloaded on the first run.")


if __name__ == "__main__":
    main()

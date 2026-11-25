"""Classify customer complaints and report per-class precision, recall, and F1."""

import random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(63)
torch.set_num_threads(1)
randomizer = random.Random(63)
categories = {
    "billing": [
        "billing charge seems incorrect",
        "payment refund was missing",
        "charged twice for purchase",
    ],
    "technical": [
        "mobile app keeps crashing",
        "login page is broken",
        "network connection is failing",
    ],
    "delivery": [
        "package delivery was delayed",
        "order arrived much later",
        "shipment status is missing",
    ],
}
class_names = list(categories)
examples = []
for label, name in enumerate(class_names):
    for _ in range(180):
        words = randomizer.choice(categories[name]).split()
        examples.append((words, label))
randomizer.shuffle(examples)
vocabulary = {"<pad>": 0}
for words, _ in examples:
    for word in words:
        if word not in vocabulary:
            vocabulary[word] = len(vocabulary)
features = torch.tensor([[vocabulary[word] for word in words] for words, _ in examples])
labels = torch.tensor([label for _, label in examples], dtype=torch.long)
split = int(0.8 * len(labels))
train_loader = DataLoader(
    TensorDataset(features[:split], labels[:split]), batch_size=32, shuffle=True
)


class ComplaintRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocabulary), 20, padding_idx=0)
        self.rnn = nn.RNN(20, 28, batch_first=True)
        self.output = nn.Linear(28, len(class_names))

    def forward(self, message):
        _, state = self.rnn(self.embedding(message))
        return self.output(state[-1])


model = ComplaintRNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for _ in range(24):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
with torch.no_grad():
    predictions = model(features[split:]).argmax(1)
    actual = labels[split:]
for label, name in enumerate(class_names):
    true_positive = ((predictions == label) & (actual == label)).sum().item()
    false_positive = ((predictions == label) & (actual != label)).sum().item()
    false_negative = ((predictions != label) & (actual == label)).sum().item()
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    print(f"{name:9}: precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}")
print(f"Overall accuracy: {(predictions == actual).float().mean().item():.3f}")

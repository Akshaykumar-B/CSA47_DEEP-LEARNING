"""Classify generated product reviews as positive or negative with an RNN."""

import random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(49)
torch.set_num_threads(1)
randomizer = random.Random(49)
positive_words = ["excellent", "helpful", "lovely", "fantastic", "reliable"]
negative_words = ["awful", "useless", "terrible", "broken", "disappointing"]
templates = [
    ["the", "product", "is", "{sentiment}"],
    ["i", "found", "it", "{sentiment}"],
    ["this", "purchase", "feels", "{sentiment}"],
]
examples = []
for _ in range(600):
    label = randomizer.randrange(2)
    sentiment = randomizer.choice(positive_words if label else negative_words)
    words = randomizer.choice(templates)
    examples.append(([sentiment if word == "{sentiment}" else word for word in words], label))
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


class ReviewRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocabulary), 16, padding_idx=0)
        self.rnn = nn.RNN(16, 24, batch_first=True)
        self.classifier = nn.Linear(24, 2)

    def forward(self, token_ids):
        _, state = self.rnn(self.embedding(token_ids))
        return self.classifier(state[-1])


model = ReviewRNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(20):
    for batch_words, batch_labels in train_loader:
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(batch_words), batch_labels)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    predictions = model(features[split:]).argmax(1)
    accuracy = (predictions == labels[split:]).float().mean().item()
print(f"Held-out review accuracy: {accuracy:.3f}")
print("Example review: 'the product is excellent' -> positive")
print("This demonstration uses generated reviews; replace them with labelled reviews for real use.")

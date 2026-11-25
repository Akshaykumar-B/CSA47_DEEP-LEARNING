"""Train a binary recursive neural network on small sentiment trees."""

import random

import torch
from torch import nn


torch.manual_seed(64)
torch.set_num_threads(1)
randomizer = random.Random(64)


def combine(left, right):
    return (left, right)


positive = ["good", "great", "excellent", "pleasant"]
negative = ["bad", "awful", "poor", "dreadful"]
nouns = ["movie", "meal", "service", "story"]
adverbs = ["very", "really", "quite"]
examples = []
for _ in range(400):
    adjective = randomizer.choice(positive + negative)
    base_label = int(adjective in positive)
    negated = randomizer.random() < 0.3
    modifier = randomizer.choice(adverbs) if randomizer.random() < 0.6 else "plain"
    subject = combine(randomizer.choice(["the", "this"]), randomizer.choice(nouns))
    adjective_phrase = combine(modifier, adjective)
    if negated:
        adjective_phrase = combine("not", adjective_phrase)
    predicate = combine("is", adjective_phrase)
    tree = combine(subject, predicate)
    examples.append((tree, base_label ^ int(negated)))
randomizer.shuffle(examples)
vocabulary = sorted({
    "the", "this", *nouns, "is", "plain", "not", *adverbs, *positive, *negative
})
word_to_id = {word: index for index, word in enumerate(vocabulary)}
split = int(0.8 * len(examples))
train_examples = examples[:split]


class TreeRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocabulary), 20)
        self.left = nn.Linear(20, 20, bias=False)
        self.right = nn.Linear(20, 20, bias=False)
        self.bias = nn.Parameter(torch.zeros(20))
        self.classifier = nn.Linear(20, 2)

    def encode(self, node):
        if isinstance(node, str):
            token = torch.tensor(word_to_id[node])
            return self.embedding(token)
        left_state = self.encode(node[0])
        right_state = self.encode(node[1])
        return torch.tanh(self.left(left_state) + self.right(right_state) + self.bias)

    def forward(self, tree):
        return self.classifier(self.encode(tree))


model = TreeRNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
for epoch in range(35):
    randomizer.shuffle(train_examples)
    for start in range(0, len(train_examples), 32):
        batch = train_examples[start:start + 32]
        logits = torch.stack([model(tree) for tree, _ in batch])
        labels = torch.tensor([label for _, label in batch], dtype=torch.long)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
with torch.no_grad():
    correct = sum(
        model(tree).argmax().item() == label for tree, label in examples[split:]
    )
print(f"Held-out tree-sentiment accuracy: {correct / (len(examples) - split):.3f}")
print("A binary tree composes subject and predicate representations recursively.")

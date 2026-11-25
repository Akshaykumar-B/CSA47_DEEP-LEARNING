"""Train a word-level RNN to predict the next word in a message."""

import torch
from torch import nn


torch.manual_seed(53)
torch.set_num_threads(1)
sentences = [
    "the customer likes friendly service",
    "the customer needs quick support",
    "shoppers prefer simple checkout",
    "people want reliable delivery",
    "the customer likes helpful answers",
    "shoppers need fast delivery",
]
words = " ".join(sentences * 40).lower().split()
vocabulary = {word: index for index, word in enumerate(sorted(set(words)))}
encoded = torch.tensor([vocabulary[word] for word in words], dtype=torch.long)
context_size = 3
contexts = torch.stack([
    encoded[index:index + context_size]
    for index in range(len(encoded) - context_size)
])
targets = encoded[context_size:]
split = int(0.8 * len(contexts))
x_train, y_train = contexts[:split], targets[:split]
x_test, y_test = contexts[split:], targets[split:]


class NextWordRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocabulary), 20)
        self.rnn = nn.RNN(20, 32, batch_first=True)
        self.output = nn.Linear(32, len(vocabulary))

    def forward(self, context):
        states, _ = self.rnn(self.embedding(context))
        return self.output(states[:, -1])


model = NextWordRNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(30):
    permutation = torch.randperm(len(x_train))
    for indices in permutation.split(64):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x_train[indices]), y_train[indices])
        loss.backward()
        optimizer.step()
    if epoch in (0, 14, 29):
        print(f"Epoch {epoch + 1}: next-word loss={loss.item():.3f}")
with torch.no_grad():
    accuracy = (model(x_test).argmax(1) == y_test).float().mean().item()
print(f"Held-out next-word accuracy: {accuracy:.3f}")
example = ["the", "customer", "likes"]
input_ids = torch.tensor([[vocabulary[word] for word in example]])
with torch.no_grad():
    prediction = max(vocabulary, key=lambda word: model(input_ids)[0, vocabulary[word]].item())
print(f"'{ ' '.join(example) }' -> '{prediction}'")

"""Train a character-level RNN and generate text from a short corpus."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(50)
torch.set_num_threads(1)
corpus = (
    "recurrent networks learn patterns from ordered characters. "
    "A hidden state carries information from earlier parts of a sequence. "
) * 8
characters = sorted(set(corpus))
to_id = {character: index for index, character in enumerate(characters)}
encoded = torch.tensor([to_id[character] for character in corpus], dtype=torch.long)
window = 24
training_end = int(0.85 * (len(encoded) - 1))
train_inputs = torch.stack([encoded[index:index + window] for index in range(training_end - window)])
train_targets = torch.stack(
    [encoded[index + 1:index + window + 1] for index in range(training_end - window)]
)
loader = DataLoader(TensorDataset(train_inputs, train_targets), batch_size=64, shuffle=True)
test_inputs = torch.stack([
    encoded[index:index + window]
    for index in range(training_end - window, len(encoded) - window)
])
test_targets = torch.stack([
    encoded[index + 1:index + window + 1]
    for index in range(training_end - window, len(encoded) - window)
])


class CharacterRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(characters), 20)
        self.rnn = nn.RNN(20, 40, batch_first=True)
        self.output = nn.Linear(40, len(characters))

    def forward(self, sequence):
        states, _ = self.rnn(self.embedding(sequence))
        return self.output(states)


model = CharacterRNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(16):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, len(characters)), batch_y.reshape(-1))
        loss.backward()
        optimizer.step()
    if epoch in (0, 7, 15):
        print(f"Epoch {epoch + 1}: character loss={loss.item():.3f}")

with torch.no_grad():
    test_logits = model(test_inputs)
    accuracy = (test_logits.argmax(-1) == test_targets).float().mean().item()
    generated = list("recurrent ")
    for _ in range(160):
        context = torch.tensor([[to_id[character] for character in generated[-window:]]])
        next_id = model(context)[0, -1].argmax().item()
        generated.append(characters[next_id])
print(f"Held-out next-character accuracy: {accuracy:.3f}")
print("Generated text:\n", "".join(generated))

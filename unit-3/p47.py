"""Use a simple RNN to classify temporal patterns in machine sensor readings."""

import torch
from sklearn.model_selection import train_test_split
from torch import nn


torch.manual_seed(47)
torch.set_num_threads(1)
count, steps = 900, 30
time_axis = torch.linspace(0, 2 * torch.pi, steps)
labels = torch.randint(0, 2, (count,))
signals = torch.sin(time_axis).repeat(count, 1)
signals += (labels.float() * 0.8 - 0.4).unsqueeze(1) * torch.linspace(0, 1, steps)
signals += torch.randn(count, steps) * 0.12
sequences = signals.unsqueeze(-1)
train_indices, test_indices = train_test_split(
    range(count), test_size=0.25, random_state=47, stratify=labels.numpy()
)
x_train, y_train = sequences[train_indices], labels[train_indices]
x_test, y_test = sequences[test_indices], labels[test_indices]


class SensorRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=16, batch_first=True)
        self.output = nn.Linear(16, 2)

    def forward(self, sequence):
        _, final_state = self.rnn(sequence)
        return self.output(final_state[-1])


model = SensorRNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(30):
    optimizer.zero_grad()
    loss = nn.functional.cross_entropy(model(x_train), y_train)
    loss.backward()
    optimizer.step()
    if epoch in (0, 9, 19, 29):
        print(f"Epoch {epoch + 1}: training loss={loss.item():.4f}")
with torch.no_grad():
    accuracy = (model(x_test).argmax(1) == y_test).float().mean().item()
print(f"Sensor-pattern test accuracy: {accuracy:.3f}")

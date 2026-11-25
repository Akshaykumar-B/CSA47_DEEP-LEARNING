"""Use a two-layer RNN to forecast multivariate machine sensor readings."""

import torch
from torch import nn


torch.manual_seed(58)
torch.set_num_threads(1)
time = torch.arange(750, dtype=torch.float32)
temperature = 65 + 0.018 * time + 2.5 * torch.sin(2 * torch.pi * time / 50)
pressure = 30 + 1.4 * torch.sin(2 * torch.pi * time / 33) + 0.002 * time
vibration = 4 + 0.5 * torch.sin(2 * torch.pi * time / 9) + 0.001 * time
sensors = torch.stack([temperature, pressure, vibration], dim=1)
sensors += torch.randn_like(sensors) * torch.tensor([0.3, 0.2, 0.08])
window = 25
split = int(0.8 * len(sensors))
mean, scale = sensors[:split].mean(0), sensors[:split].std(0)
normalized = (sensors - mean) / scale
sequences = torch.stack([normalized[i:i + window] for i in range(len(sensors) - window)])
targets = normalized[window:]
train_count = split - window
x_train, y_train = sequences[:train_count], targets[:train_count]
x_test, y_test = sequences[train_count:], targets[train_count:]


class StackedSensorRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(3, 32, num_layers=2, batch_first=True, dropout=0.1)
        self.output = nn.Linear(32, 3)

    def forward(self, sequence):
        states, _ = self.rnn(sequence)
        return self.output(states[:, -1])


model = StackedSensorRNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
for epoch in range(70):
    model.train()
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(x_train), y_train)
    loss.backward()
    optimizer.step()
    if epoch in (0, 19, 44, 69):
        print(f"Epoch {epoch + 1}: normalized training MSE={loss.item():.4f}")
model.eval()
with torch.no_grad():
    predictions = model(x_test) * scale + mean
    actual = y_test * scale + mean
    mae = (predictions - actual).abs().mean(0)
for name, value in zip(("temperature", "pressure", "vibration"), mae):
    print(f"Held-out {name} MAE: {value.item():.3f}")

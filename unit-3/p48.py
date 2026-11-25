"""Predict the next daily temperature from a rolling window with an RNN."""

import torch
from torch import nn


torch.manual_seed(48)
torch.set_num_threads(1)
days = torch.arange(0, 620, dtype=torch.float32)
temperatures = (
    18
    + 9 * torch.sin(2 * torch.pi * days / 365.0)
    + 0.004 * days
    + torch.randn(len(days)) * 0.8
)
window = 30
sequences = torch.stack([temperatures[i:i + window] for i in range(len(days) - window)])
targets = temperatures[window:].unsqueeze(1)
split = int(0.8 * len(sequences))
x_train, x_test = sequences[:split], sequences[split:]
y_train, y_test = targets[:split], targets[split:]
mean, std = x_train.mean(), x_train.std()
x_train = ((x_train - mean) / std).unsqueeze(-1)
x_test = ((x_test - mean) / std).unsqueeze(-1)
y_train = (y_train - mean) / std


class TemperatureRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=24, batch_first=True)
        self.output = nn.Linear(24, 1)

    def forward(self, sequence):
        states, _ = self.rnn(sequence)
        return self.output(states[:, -1])


model = TemperatureRNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(80):
    model.train()
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(x_train), y_train)
    loss.backward()
    optimizer.step()
    if epoch in (0, 19, 39, 59, 79):
        print(f"Epoch {epoch + 1}: normalized training MSE={loss.item():.4f}")
model.eval()
with torch.no_grad():
    predictions = model(x_test).squeeze(1) * std + mean
    actual = y_test.squeeze(1)
    mae = (predictions - actual).abs().mean().item()
print(f"Held-out MAE: {mae:.2f} degrees")
print("First five actual / predicted temperatures:")
for actual_value, predicted_value in zip(actual[:5], predictions[:5]):
    print(f"{actual_value.item():.1f} / {predicted_value.item():.1f}")

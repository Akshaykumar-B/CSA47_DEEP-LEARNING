"""Evaluate electricity-demand RNN forecasts with MAE, MSE, and RMSE."""

import torch
from torch import nn


torch.manual_seed(62)
torch.set_num_threads(1)
day = torch.arange(800, dtype=torch.float32)
consumption = 150 + 17 * torch.sin(2 * torch.pi * day / 7)
consumption += 22 * torch.sin(2 * torch.pi * day / 180) + 0.015 * day
consumption += torch.randn(len(day)) * 2.5
window = 21
split = int(0.8 * len(consumption))
mean, scale = consumption[:split].mean(), consumption[:split].std()
normalized = (consumption - mean) / scale
sequences = torch.stack([normalized[i:i + window] for i in range(len(consumption) - window)])
targets = normalized[window:].unsqueeze(1)
train_count = split - window
x_train, y_train = sequences[:train_count].unsqueeze(-1), targets[:train_count]
x_test, y_test = sequences[train_count:].unsqueeze(-1), targets[train_count:]


class ConsumptionRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 24, batch_first=True)
        self.output = nn.Linear(24, 1)

    def forward(self, sequence):
        states, _ = self.rnn(sequence)
        return self.output(states[:, -1])


model = ConsumptionRNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for _ in range(60):
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(x_train), y_train)
    loss.backward()
    optimizer.step()
with torch.no_grad():
    forecast = model(x_test).squeeze(1) * scale + mean
    actual = y_test.squeeze(1) * scale + mean
    errors = forecast - actual
mae = errors.abs().mean().item()
mse = errors.square().mean().item()
rmse = errors.square().mean().sqrt().item()
print(f"MAE:  {mae:.3f} consumption units")
print(f"MSE:  {mse:.3f} units squared")
print(f"RMSE: {rmse:.3f} consumption units")

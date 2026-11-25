"""Compare RNN temperature forecasts using different history-window lengths."""

import torch
from torch import nn


torch.manual_seed(60)
torch.set_num_threads(1)
day = torch.arange(620, dtype=torch.float32)
temperature = 16 + 8 * torch.sin(2 * torch.pi * day / 90)
temperature += 2 * torch.sin(2 * torch.pi * day / 17) + torch.randn(len(day)) * 0.45
split = int(0.8 * len(temperature))
mean, scale = temperature[:split].mean(), temperature[:split].std()
normalized = (temperature - mean) / scale


class TemperatureRNN(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.rnn = nn.RNN(1, 24, batch_first=True)
        self.output = nn.Linear(24, 1)

    def forward(self, sequence):
        states, _ = self.rnn(sequence)
        return self.output(states[:, -1])


for window in (7, 14, 30, 60):
    sequences = torch.stack([
        normalized[index:index + window]
        for index in range(len(temperature) - window)
    ]).unsqueeze(-1)
    targets = normalized[window:].unsqueeze(1)
    train_count = split - window
    x_train, y_train = sequences[:train_count], targets[:train_count]
    x_test, y_test = sequences[train_count:], targets[train_count:]
    model = TemperatureRNN(window)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(50):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        forecast = model(x_test).squeeze(1) * scale + mean
        actual = y_test.squeeze(1) * scale + mean
        mae = (forecast - actual).abs().mean().item()
    print(f"History window={window:2} days: held-out MAE={mae:.3f}")
print("A longer window adds context but may also add noise and training cost.")

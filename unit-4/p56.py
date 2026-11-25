"""Use a GRU to forecast multiple synthetic weather measurements."""

import torch
from torch import nn


torch.manual_seed(56)
torch.set_num_threads(1)
day = torch.arange(800, dtype=torch.float32)
temperature = 18 + 10 * torch.sin(2 * torch.pi * day / 365) + torch.randn(len(day)) * 0.7
humidity = 62 - 13 * torch.sin(2 * torch.pi * day / 365) + torch.randn(len(day)) * 2
pressure = 1013 + 4 * torch.sin(2 * torch.pi * day / 18) + torch.randn(len(day)) * 0.8
weather = torch.stack([temperature, humidity, pressure], dim=1)
window = 30
split = int(0.8 * len(weather))
mean, scale = weather[:split].mean(0), weather[:split].std(0)
normalized = (weather - mean) / scale
sequences = torch.stack([normalized[i:i + window] for i in range(len(weather) - window)])
targets = normalized[window:]
train_count = split - window
x_train, y_train = sequences[:train_count], targets[:train_count]
x_test, y_test = sequences[train_count:], targets[train_count:]


class WeatherGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(3, 32, batch_first=True)
        self.output = nn.Linear(32, 3)

    def forward(self, sequence):
        states, _ = self.gru(sequence)
        return self.output(states[:, -1])


model = WeatherGRU()
optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
for epoch in range(70):
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(x_train), y_train)
    loss.backward()
    optimizer.step()
    if epoch in (0, 19, 44, 69):
        print(f"Epoch {epoch + 1}: normalized MSE={loss.item():.4f}")
with torch.no_grad():
    prediction = model(x_test) * scale + mean
    actual = y_test * scale + mean
    mae_by_measurement = (prediction - actual).abs().mean(0)
for name, mae in zip(("temperature", "humidity", "pressure"), mae_by_measurement):
    print(f"Held-out {name} MAE: {mae.item():.2f}")

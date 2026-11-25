"""Compare RNN and LSTM demand forecasting on identical time-series windows."""

import torch
from torch import nn


torch.manual_seed(54)
torch.set_num_threads(1)
time = torch.arange(650, dtype=torch.float32)
demand = 40 + 7 * torch.sin(2 * torch.pi * time / 30) + 2 * torch.sin(2 * torch.pi * time / 7)
demand += 0.008 * time + torch.randn(len(time)) * 0.5
window = 24
split = int(0.8 * len(demand))
mean, scale = demand[:split].mean(), demand[:split].std()
normalized = (demand - mean) / scale
sequences = torch.stack([normalized[i:i + window] for i in range(len(demand) - window)])
targets = normalized[window:].unsqueeze(1)
train_count = split - window
x_train, y_train = sequences[:train_count].unsqueeze(-1), targets[:train_count]
x_test, y_test = sequences[train_count:].unsqueeze(-1), targets[train_count:]


class Forecaster(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.recurrent = architecture(1, 24, batch_first=True)
        self.output = nn.Linear(24, 1)

    def forward(self, sequence):
        result = self.recurrent(sequence)
        states = result[0]
        return self.output(states[:, -1])


for name, architecture in (("RNN", nn.RNN), ("LSTM", nn.LSTM)):
    model = Forecaster(architecture)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(60):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = model(x_test).squeeze(1) * scale + mean
        actual = y_test.squeeze(1) * scale + mean
        mae = (prediction - actual).abs().mean().item()
        rmse = ((prediction - actual).square().mean().sqrt()).item()
    print(f"{name:4}: held-out MAE={mae:.3f}, RMSE={rmse:.3f}")
print("Both architectures use the same demand series, split, and input-window length.")

"""Forecast synthetic stock prices with an RNN and report regression errors."""

import torch
from torch import nn


torch.manual_seed(51)
torch.set_num_threads(1)
time = torch.arange(720, dtype=torch.float32)
prices = 100 + 0.035 * time + 2.8 * torch.sin(2 * torch.pi * time / 90)
prices += 0.7 * torch.sin(2 * torch.pi * time / 17) + torch.randn(len(time)) * 0.45
window = 30
split = int(0.8 * len(prices))
mean, scale = prices[:split].mean(), prices[:split].std()
normalized = (prices - mean) / scale
sequences = torch.stack([normalized[i:i + window] for i in range(len(prices) - window)])
targets = normalized[window:].unsqueeze(1)
train_count = split - window
x_train, y_train = sequences[:train_count].unsqueeze(-1), targets[:train_count]
x_test, y_test = sequences[train_count:].unsqueeze(-1), targets[train_count:]


class PriceRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 24, batch_first=True)
        self.output = nn.Linear(24, 1)

    def forward(self, sequence):
        states, _ = self.rnn(sequence)
        return self.output(states[:, -1])


model = PriceRNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(60):
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(x_train), y_train)
    loss.backward()
    optimizer.step()
    if epoch in (0, 19, 39, 59):
        print(f"Epoch {epoch + 1}: training MSE={loss.item():.4f}")
with torch.no_grad():
    predicted = model(x_test).squeeze(1) * scale + mean
    actual = y_test.squeeze(1) * scale + mean
    mae = (predicted - actual).abs().mean().item()
    rmse = ((predicted - actual).square().mean().sqrt()).item()
print(f"Held-out MAE: {mae:.2f}; RMSE: {rmse:.2f}")
print("Synthetic prices are used here; use historical market data for real forecasting.")

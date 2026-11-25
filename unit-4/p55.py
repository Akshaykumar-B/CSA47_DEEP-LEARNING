"""Forecast a synthetic stock-price series with an LSTM."""

import torch
from torch import nn


torch.manual_seed(55)
torch.set_num_threads(1)
time = torch.arange(720, dtype=torch.float32)
price = 75 + 0.025 * time + 3 * torch.sin(2 * torch.pi * time / 110)
price += 0.8 * torch.sin(2 * torch.pi * time / 23) + torch.randn(len(time)) * 0.5
window = 35
split = int(0.8 * len(price))
mean, scale = price[:split].mean(), price[:split].std()
normalized = (price - mean) / scale
sequences = torch.stack([normalized[i:i + window] for i in range(len(price) - window)])
targets = normalized[window:].unsqueeze(1)
train_count = split - window
x_train, y_train = sequences[:train_count].unsqueeze(-1), targets[:train_count]
x_test, y_test = sequences[train_count:].unsqueeze(-1), targets[train_count:]


class StockLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.output = nn.Linear(32, 1)

    def forward(self, sequence):
        states, _ = self.lstm(sequence)
        return self.output(states[:, -1])


model = StockLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
for epoch in range(70):
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(x_train), y_train)
    loss.backward()
    optimizer.step()
    if epoch in (0, 19, 44, 69):
        print(f"Epoch {epoch + 1}: normalized MSE={loss.item():.4f}")
with torch.no_grad():
    prediction = model(x_test).squeeze(1) * scale + mean
    actual = y_test.squeeze(1) * scale + mean
    errors = prediction - actual
print(f"MAE={errors.abs().mean().item():.2f}, MSE={errors.square().mean().item():.2f}, RMSE={errors.square().mean().sqrt().item():.2f}")
print("This uses a synthetic price series, not investment data or advice.")

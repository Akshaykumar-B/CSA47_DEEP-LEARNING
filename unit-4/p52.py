"""Forecast synthetic daily electricity demand from historical observations."""

import torch
from torch import nn


torch.manual_seed(52)
torch.set_num_threads(1)
days = torch.arange(900, dtype=torch.float32)
demand = (
    120
    + 18 * torch.sin(2 * torch.pi * days / 7)
    + 24 * torch.sin(2 * torch.pi * days / 365)
    + 0.012 * days
    + torch.randn(len(days)) * 3
)
window = 28
split = int(0.8 * len(demand))
mean, scale = demand[:split].mean(), demand[:split].std()
normalized = (demand - mean) / scale
sequences = torch.stack([normalized[i:i + window] for i in range(len(demand) - window)])
targets = normalized[window:].unsqueeze(1)
train_count = split - window
x_train, y_train = sequences[:train_count].unsqueeze(-1), targets[:train_count]
x_test, y_test = sequences[train_count:].unsqueeze(-1), targets[train_count:]


class DemandRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 24, batch_first=True)
        self.output = nn.Linear(24, 1)

    def forward(self, sequence):
        states, _ = self.rnn(sequence)
        return self.output(states[:, -1])


model = DemandRNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(60):
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(x_train), y_train)
    loss.backward()
    optimizer.step()
    if epoch in (0, 19, 39, 59):
        print(f"Epoch {epoch + 1}: training MSE={loss.item():.4f}")
with torch.no_grad():
    forecast = model(x_test).squeeze(1) * scale + mean
    actual = y_test.squeeze(1) * scale + mean
print("First five actual / predicted daily demand values:")
for actual_value, predicted_value in zip(actual[:5], forecast[:5]):
    print(f"{actual_value.item():.1f} / {predicted_value.item():.1f}")
print(f"Test MAE: {(forecast - actual).abs().mean().item():.2f}")

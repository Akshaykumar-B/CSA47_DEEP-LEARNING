"""Compare RNN, LSTM, and GRU on one synthetic forecasting problem."""

import torch
from torch import nn


torch.manual_seed(57)
torch.set_num_threads(1)
time = torch.arange(700, dtype=torch.float32)
series = 8 + 1.8 * torch.sin(2 * torch.pi * time / 75) + 0.6 * torch.sin(2 * torch.pi * time / 13)
series += torch.randn(len(time)) * 0.18
window = 28
split = int(0.8 * len(series))
mean, scale = series[:split].mean(), series[:split].std()
normalized = (series - mean) / scale
sequences = torch.stack([normalized[i:i + window] for i in range(len(series) - window)])
targets = normalized[window:].unsqueeze(1)
train_count = split - window
x_train, y_train = sequences[:train_count].unsqueeze(-1), targets[:train_count]
x_test, y_test = sequences[train_count:].unsqueeze(-1), targets[train_count:]


class SequenceModel(nn.Module):
    def __init__(self, recurrent_type):
        super().__init__()
        self.recurrent = recurrent_type(1, 24, batch_first=True)
        self.output = nn.Linear(24, 1)

    def forward(self, sequence):
        states, _ = self.recurrent(sequence)
        return self.output(states[:, -1])


for name, recurrent_type in (("RNN", nn.RNN), ("LSTM", nn.LSTM), ("GRU", nn.GRU)):
    model = SequenceModel(recurrent_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(55):
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
print("All three models use identical train/test windows and training settings.")

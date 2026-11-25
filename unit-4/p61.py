"""Compare LSTM demand-forecasting hyperparameter configurations."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(61)
torch.set_num_threads(1)
time = torch.arange(520, dtype=torch.float32)
demand = 35 + 6 * torch.sin(2 * torch.pi * time / 28) + 1.5 * torch.sin(2 * torch.pi * time / 7)
demand += torch.randn(len(time)) * 0.35
window = 20
split = int(0.8 * len(demand))
mean, scale = demand[:split].mean(), demand[:split].std()
normalized = (demand - mean) / scale
sequences = torch.stack([normalized[i:i + window] for i in range(len(demand) - window)])
targets = normalized[window:].unsqueeze(1)
train_count = split - window
x_train, y_train = sequences[:train_count].unsqueeze(-1), targets[:train_count]
x_test, y_test = sequences[train_count:].unsqueeze(-1), targets[train_count:]
configurations = [
    (0.01, 16, 16),
    (0.003, 16, 32),
    (0.01, 32, 32),
    (0.003, 32, 64),
    (0.02, 48, 16),
    (0.01, 48, 64),
]


class DemandLSTM(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_units, batch_first=True)
        self.output = nn.Linear(hidden_units, 1)

    def forward(self, sequence):
        states, _ = self.lstm(sequence)
        return self.output(states[:, -1])


results = []
for learning_rate, hidden_units, batch_size in configurations:
    model = DemandLSTM(hidden_units)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    for _ in range(24):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(x_test).squeeze(1) * scale + mean
        actual = y_test.squeeze(1) * scale + mean
        mae = (prediction - actual).abs().mean().item()
    results.append((mae, learning_rate, hidden_units, batch_size))
    print(
        f"lr={learning_rate:g}, hidden={hidden_units:2}, batch={batch_size:2}: "
        f"test MAE={mae:.3f}"
    )
best = min(results)
print(f"Best tested configuration: lr={best[1]:g}, hidden={best[2]}, batch={best[3]}")

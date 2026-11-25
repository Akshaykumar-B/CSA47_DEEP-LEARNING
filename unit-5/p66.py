"""Compare RNN and LSTM memory on a long-range sequence classification task."""

import torch
from torch import nn


torch.manual_seed(66)
torch.set_num_threads(1)
sequence_length = 60


def make_data(count):
    labels = torch.randint(0, 2, (count,))
    sequences = torch.randn(count, sequence_length, 1) * 0.6
    sequences[:, 0, 0] = labels.float() * 2 - 1
    return sequences, labels


torch.manual_seed(68)
x_train, y_train = make_data(2000)
x_test, y_test = make_data(500)


class SequenceClassifier(nn.Module):
    def __init__(self, cell_type):
        super().__init__()
        self.recurrent = cell_type(1, 24, batch_first=True)
        self.classifier = nn.Linear(24, 2)

    def forward(self, sequence):
        result = self.recurrent(sequence)
        states = result[0]
        return self.classifier(states[:, -1])


for name, cell in (("RNN", nn.RNN), ("LSTM", nn.LSTM)):
    torch.manual_seed(70)
    model = SequenceClassifier(cell)
    if name == "LSTM":
        with torch.no_grad():
            for parameter_name, parameter in model.recurrent.named_parameters():
                if parameter_name.startswith("bias_ih"):
                    parameter[24:48].fill_(3.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(100):
        model.train()
        permutation = torch.randperm(len(x_train))
        for indices in permutation.split(64):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x_train[indices]), y_train[indices])
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        accuracy = (model(x_test).argmax(1) == y_test).float().mean().item()
    print(f"{name:4}: held-out accuracy={accuracy:.3f}")
print(f"The label is encoded at time step 1; each model must retain it across {sequence_length - 1} steps.")

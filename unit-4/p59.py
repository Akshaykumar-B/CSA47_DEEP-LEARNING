"""Show how a tanh RNN's gradient can shrink across many time steps."""

import torch


torch.set_num_threads(1)
recurrent_weight = 0.55
initial_value = 0.8
checkpoints = (1, 5, 10, 20, 50, 100)
print("Steps | absolute gradient d(h_T)/d(h_0)")
for steps in checkpoints:
    initial = torch.tensor(initial_value, requires_grad=True)
    hidden = initial
    for _ in range(steps):
        hidden = torch.tanh(recurrent_weight * hidden)
    gradient = torch.autograd.grad(hidden, initial)[0].abs().item()
    print(f"{steps:5} | {gradient:.8e}")
print("Repeated derivatives below one multiply together, causing early-time gradients to vanish.")

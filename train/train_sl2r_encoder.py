import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from lie_groups import SL2R
from manifold_plotter import plot_error_trend, plot_lie_error, plot_sl2r_trajectory_comparison

SEQ_LEN = 100
DT = 0.1
EPOCHS = 1000
BATCH_SIZE = 64
LR = 1e-3

def simulate_sl2r_trajectory(xi, seq_len, dt):
    g = SL2R.exp(np.zeros(3))
    traj = []
    for _ in range(seq_len):
        delta = SL2R.exp(xi * dt)
        g = g @ delta
        g = SL2R(g.mat)
        traj.append(g)
    return traj

def trajectory_to_twist_sequence(traj):
    xi_seq = []
    prev = traj[0]
    for current in traj[1:]:
        rel = prev.inv() @ current
        xi_seq.append(rel.log())
        prev = current
    return np.stack(xi_seq)

class SL2REncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)

x_data, y_data = [], []
for _ in range(2048):
    xi = np.random.uniform(-10, 10, size=3)
    norm = np.linalg.norm(xi)
    if norm > 1.5:
        xi = xi / norm * 1.5 
    traj = simulate_sl2r_trajectory(xi, SEQ_LEN, DT)
    xi_seq = trajectory_to_twist_sequence(traj)
    x_data.append(xi_seq)
    y_data.append(xi)

x_tensor = torch.tensor(np.array(x_data), dtype=torch.float32)
y_tensor = torch.tensor(np.array(y_data), dtype=torch.float32)

mean = x_tensor.mean(dim=(0, 1), keepdim=True)
std = x_tensor.std(dim=(0, 1), keepdim=True) + 1e-6
x_norm = (x_tensor - mean) / std

model = SL2REncoder(input_dim=3 * (SEQ_LEN - 1))
opt = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

error_log = []

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(x_tensor))
    err_epoch = []

    for i in range(0, len(perm), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        xb = x_norm[idx]
        yb = y_tensor[idx]
        xb = xb.view(xb.size(0), -1)

        pred = model(xb)
        loss = loss_fn(pred, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        err_epoch.append(torch.abs(pred - yb).detach().cpu().numpy())

    avg_err = np.mean(np.vstack(err_epoch), axis=0)
    error_log.append(avg_err)
    print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.6f} | Error = {avg_err}")

model.eval()
x_sample = x_norm[0:1].view(1, -1)
pred = model(x_sample).detach().numpy().squeeze()
true = y_tensor[0].numpy()

true_traj = simulate_sl2r_trajectory(true, SEQ_LEN, DT)
pred_traj = simulate_sl2r_trajectory(pred, SEQ_LEN, DT)

plot_sl2r_trajectory_comparison(true_traj, pred_traj)
print("True:", true)
print("Pred:", pred)
print("Absolute Errors:", np.abs(pred - true))

plot_error_trend(error_log, labels=["a", "b", "c"])
plot_lie_error(true, pred, title="SL(2,R) Lie Algebra Error")

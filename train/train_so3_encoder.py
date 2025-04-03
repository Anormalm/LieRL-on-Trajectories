import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from lie_groups import SO3
from manifold_plotter import (
    plot_error_trend, 
    plot_lie_error, 
    plot_so3_orientation_evolution,
    plot_so3_trajectory_comparison
)

SEQ_LEN = 100
DT = 0.1
EPOCHS = 1000
BATCH_SIZE = 64
LR = 1e-3
NOISE_STD = 0.0  

def simulate_so3_trajectory(omega, seq_len, dt):
    """Generates an SO(3) trajectory from angular velocity."""
    R = SO3.exp(np.zeros(3)) 
    trajectory = []
    for _ in range(seq_len):
        delta = SO3.exp(omega * dt)
        R = R @ delta
        trajectory.append(R)
    return trajectory

def sample_oblique_omega():
    scale = np.random.uniform(1, 10)
    direction = np.random.randn(3)
    direction /= np.linalg.norm(direction) + 1e-8
    return scale * direction

def trajectory_to_twist_sequence(traj):
    """Convert SO(3) pose sequence to Lie algebra velocity sequence."""
    xi_seq = []
    prev = traj[0]
    for curr in traj[1:]:
        rel = prev.inv() @ curr
        xi_seq.append(rel.log())
        prev = curr
    return np.stack(xi_seq)

class SO3Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * (SEQ_LEN - 1), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)

num_samples = 2048
x_data, y_labels = [], []

for _ in range(num_samples):
    omega = sample_oblique_omega()
    poses = simulate_so3_trajectory(omega, SEQ_LEN, DT)
    xi_seq = trajectory_to_twist_sequence(poses)
    x_data.append(xi_seq)
    y_labels.append(omega)

x_tensor = torch.tensor(np.array(x_data), dtype=torch.float32)
y_tensor = torch.tensor(np.array(y_labels), dtype=torch.float32)

mean = x_tensor.mean(dim=(0, 1), keepdim=True)
std = x_tensor.std(dim=(0, 1), keepdim=True) + 1e-6
x_norm = (x_tensor - mean) / std

model = SO3Encoder()
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
error_log = []

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(x_tensor))
    epoch_err = []

    for i in range(0, len(perm), BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        xb, yb = x_norm[idx], y_tensor[idx]

        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        abs_err = torch.abs(pred - yb).detach().cpu().numpy()
        epoch_err.append(abs_err)

    avg_err = np.mean(np.vstack(epoch_err), axis=0)
    error_log.append(avg_err)
    print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.6f} | Error = {avg_err}")

model.eval()
x_sample = x_norm[0:1]
true_omega = y_tensor[0].cpu().numpy()
pred_omega = model(x_sample).detach().cpu().numpy().squeeze()

true_traj = simulate_so3_trajectory(true_omega, SEQ_LEN, DT)
pred_traj = simulate_so3_trajectory(pred_omega, SEQ_LEN, DT)

plot_so3_trajectory_comparison(true_traj, pred_traj, title="SO(3) Rotation Axis Comparison")

plot_so3_orientation_evolution(true_traj, title="True Orientation Evolution (Oblique Spin)")
plt.savefig("so3_true_orientation.png", dpi=300, bbox_inches='tight')
plt.close()

plot_so3_orientation_evolution(pred_traj, title="Predicted Orientation Evolution (Oblique Spin)")
plt.savefig("so3_pred_orientation.png", dpi=300, bbox_inches='tight')
plt.close()

plot_error_trend(error_log, labels=["ω_x", "ω_y", "ω_z"])
plt.savefig("so3_error_trend.png", dpi=300, bbox_inches='tight')
plt.close()

plot_lie_error(true_omega, pred_omega, title="SO(3) Lie Algebra Error")
plt.savefig("so3_lie_error.png", dpi=300, bbox_inches='tight')
plt.close()

print("Oblique Spin Test:")
print("True ω:", true_omega)
print("Pred ω:", pred_omega)
print("Absolute Errors:", np.abs(pred_omega - true_omega))

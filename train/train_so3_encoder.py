import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from lie_groups import SO3  # Assume SO3 is implemented like SE3/SE2
from manifold_plotter import plot_error_trend, plot_lie_error, plot_so3_orientation_evolution

# === CONFIG ===
SEQ_LEN = 100
DT = 0.1
EPOCHS = 1000
BATCH_SIZE = 64
LR = 1e-3

# === Synthetic SO(3) Trajectory ===
def simulate_so3_trajectory(omega, seq_len, dt):
    R = SO3.exp(np.zeros(3))
    trajectory = []
    for _ in range(seq_len):
        delta = SO3.exp(omega * dt)
        R = R @ delta
        trajectory.append(R)
    return trajectory

def sample_diverse_omega():
    scale = np.random.uniform(1, 10)
    direction = np.random.randn(3)
    direction /= np.linalg.norm(direction) + 1e-8
    return scale * direction  # random spin with varied speed and direction

# === Log-pose to twist sequence ===
def trajectory_to_twist_sequence(trajectory):
    xi_seq = []
    prev = trajectory[0]
    for current in trajectory[1:]:
        rel = prev.inv() @ current
        xi = rel.log()
        xi_seq.append(xi)
        prev = current
    return np.stack(xi_seq)

# === Encoder Model ===
class SO3Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * (SEQ_LEN - 1), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Output is angular velocity omega ∈ ℝ³
        )

    def forward(self, x):
        return self.net(x)

# === Data Generation ===
num_samples = 2048
x_data, y_data = [], []

for _ in range(num_samples):
    # omega = np.random.uniform(-10, 10, size=3)
    omega = sample_diverse_omega()
    poses = simulate_so3_trajectory(omega, SEQ_LEN, DT)
    xi_seq = trajectory_to_twist_sequence(poses)
    x_data.append(xi_seq)
    y_data.append(omega)

x_tensor = torch.tensor(np.array(x_data), dtype=torch.float32)
y_tensor = torch.tensor(np.array(y_data), dtype=torch.float32)
true_traj = simulate_so3_trajectory(omega, SEQ_LEN, DT)

# === Normalize ===
mean = x_tensor.mean(dim=(0, 1), keepdim=True)
std = x_tensor.std(dim=(0, 1), keepdim=True) + 1e-6
# Convert to xi sequence and normalize
xi_seq = trajectory_to_twist_sequence(true_traj)
xi_tensor = torch.tensor(xi_seq, dtype=torch.float32).unsqueeze(0)  # shape (1, T-1, 3)
xi_norm = (xi_tensor - mean) / std
# === Training ===
model = SO3Encoder()
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
error_log = []

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(xi_norm.size(0))
    epoch_err = []

    for i in range(0, len(perm), BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        xb, yb = xi_norm[idx], y_tensor[idx]

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

# === Evaluate ===
model.eval()
pred_omega = model(xi_norm).detach().cpu().numpy().squeeze()
true_omega = y_tensor[0].cpu().numpy()

# Reconstruct predicted trajectory
true_traj = simulate_so3_trajectory(true_omega, SEQ_LEN, DT)
pred_traj = simulate_so3_trajectory(pred_omega, SEQ_LEN, DT)

print("Oblique Spin Test:")
print("True ω:", omega)
print("Pred ω:", pred_omega)
print("Absolute Errors:", np.abs(pred_omega - omega))

# Orientation evolution comparison
pred_traj = simulate_so3_trajectory(pred_omega, SEQ_LEN, DT)
plot_so3_orientation_evolution(true_traj, title="True Orientation Evolution (Oblique Spin)")
plot_so3_orientation_evolution(pred_traj, title="Predicted Orientation Evolution (Oblique Spin)")


plot_error_trend(error_log, labels=["ω_x", "ω_y", "ω_z"])
plot_lie_error(true_omega, pred_omega, title="SO(3) Lie Algebra Error")
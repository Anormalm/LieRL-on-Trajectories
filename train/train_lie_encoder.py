import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from lie_groups import SE3,SL2R  # assumes you already implemented this
from manifold_plotter import plot_error_trend, plot_lie_error, plot_se3_trajectory_comparison
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm, logm

# ===== CONFIG =====
SEQ_LEN = 120
DT = 0.1
EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# ===== Trajectory Simulation =====
from lie_groups import SE3  # make sure your SE3 class is used

def integrate_se3_trajectory(xi: np.ndarray, seq_len: int, dt: float):
    pose = SE3.exp(np.zeros(6))  # Identity pose
    traj = []

    for _ in range(seq_len):
        delta = SE3.exp(xi * dt)
        pose = pose @ delta
        traj.append(pose)

    return traj



# ===== Convert trajectory to Lie algebra sequence =====
def trajectory_to_lie_algebra_se3(traj):
    return np.array([SE3.log(pose) for pose in traj])

# ===== Simple Encoder Network =====
class LieEncoderSE3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(SEQ_LEN * 6, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        return self.net(x)

# ===== Generate Dataset =====
num_episodes = 1024
x_data = []
y_labels = []

NOISE_STD = 0.05  # Tune this value for desired noise level

for _ in range(num_episodes):
    true_xi = np.random.uniform(-1.0, 1.0, size=6)
    noisy_xi_seq = []
    pose = None
    for _ in range(SEQ_LEN):
        noise = np.random.normal(0, NOISE_STD, size=6)
        delta_xi = true_xi + noise
        delta = SE3.exp(delta_xi * DT)
        pose = pose @ delta if pose else SE3.exp(np.zeros(6))
        noisy_xi_seq.append(pose)
    xi_seq = trajectory_to_lie_algebra_se3(noisy_xi_seq)
    x_data.append(xi_seq)
    y_labels.append(true_xi)  # still use true (clean) xi as the label


x_tensor = torch.tensor(np.array(x_data), dtype=torch.float32)
y_tensor = torch.tensor(np.array(y_labels), dtype=torch.float32)

# ===== Normalize =====
global_mean = x_tensor.mean(dim=(0, 1), keepdim=True)
global_std = x_tensor.std(dim=(0, 1), keepdim=True) + 1e-6
x_norm = (x_tensor - global_mean) / global_std

# ===== Training =====
model = LieEncoderSE3()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()
error_log = []

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(x_norm.size(0))
    epoch_err = []

    for i in range(0, x_norm.size(0), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        xb = x_norm[idx]
        yb = y_tensor[idx]

        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        abs_err = torch.abs(pred - yb).detach().cpu().numpy()
        epoch_err.append(abs_err)

    avg_err = np.mean(np.vstack(epoch_err), axis=0)
    error_log.append(avg_err)
    print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.6f} | Error = {avg_err}")

# ===== Plot Error Trend =====
plot_error_trend(error_log, labels=["w_x", "w_y", "w_z", "v_x", "v_y", "v_z"])

# ===== Sample Visualisation =====
model.eval()
x_sample = x_norm[0:1]
pred_xi = model(x_sample).detach().cpu().numpy().squeeze()
true_xi = y_tensor[0].cpu().numpy()

true_traj = integrate_se3_trajectory(true_xi, SEQ_LEN, DT)
pred_traj = integrate_se3_trajectory(pred_xi, SEQ_LEN, DT)

plot_se3_trajectory_comparison(true_traj, pred_traj)

print("True:", true_xi)
print("Pred:", pred_xi)
plot_lie_error(true_xi, pred_xi, title="SE(3) Lie Algebra Error")

# Optional: save model
# torch.save(model.state_dict(), "lie_encoder_se3_noise.pth")


